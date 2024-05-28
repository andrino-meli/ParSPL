#!/usr/bin/env python3
import numpy as np
import argparse, os, json, sys, subprocess
from scipy.sparse import csc_matrix
import scipy.sparse as spa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from enum import Enum

from bfloat16 import bfloat16
from data_format import Csc, Triag
from data_format import Collist, Tile, DiagInv, Empty, SynchBuffer
from general import escape, HARTS, eprint, wprint, bprint, DotDict, dprint
from general import color_palette, DEBUG, ndarrayToCH, ndarrayToC, list2array
import general

np.random.seed(0)
CAT_CMD = 'bat'
DEBUG = False
BARRIER = f'\t\t\t__rt_seperator();\n'

def find_optimal_cuts(linsys,levels):
    raise NotImplemented("")

def verify_cuts(cuts,n):
    cuts.sort()
    assert(cuts[0] == 0)
    assert(cuts[-1] == n)
    for i in range(len(cuts)-1):
        assert(cuts[i] < cuts[i+1])

def tile_L(L,cuts):
    ''' Cut appart L and return matrix representation of tiles. '''
    # verify data
    n = L.n
    verify_cuts(cuts,n)

    numcuts = len(cuts)-1
    # tiles do not necessarilly have to be ordered
    tiles = [[None for j in range(i+1)] for i in range(numcuts)]

    def tile_from_index(row,col):
        ''' Get index of tile based on index of matrix. '''
        r = 0; c = 0
        while(row >= cuts[r+1]):
            r += 1
        while(col >= cuts[c+1]):
            c += 1
        #print(f'row,col {row},{col} are found in tile {r},{c}')
        return tiles[r][c]

    # compute cuts
    for i in range(numcuts):
        for j in range(i+1):
            rowa,rowz = (cuts[i],cuts[i+1])
            cola,colz = (cuts[j],cuts[j+1])
            tiles[i][j] = Collist(cuts[i],cuts[i+1],cuts[j],cuts[j+1])
    # distribute data
    for col in range(n):
        for i in range(L.Lp[col],L.Lp[col+1]):
            row = L.Li[i]
            tile = tile_from_index(row,col)
            tile.insert(row,col,L.Lx[i])
    return tiles

def assign_kernel_to_tile(tiles):
    print()
    bprint("\nAssigning Kernels to tiles:")
    numcuts = len(tiles)

    # decide on kernel for diagonal tiles
    for i in range(numcuts):
        triag = tiles[i][i]
        assert(isinstance(triag,Tile))
        if triag.nnz() == 0:
            tiles[i][i] = Empty(0,0,0,0)
        elif triag.density() > 0.8 or True: #TODO: make non_dense, non_empty diag kernels a thing
            print(f"DENSIFY: {triag}")
            tiles[i][i] = DiagInv(triag)
        elif triag.density() < 0.05:
            print(f"SPARSIFY: {triag}")
            raise NotImplementedError()
        else:
            eprint(f'Triag "{triag}" is neither sparse nor dense ({triag.density()*100:.1f}%). Inflating memory by inverting. Consider subcutting it.')
            tiles[i][i] = DiagInv(triag)
            #raise NotImplementedError()

    # decide on kernel for the rest
    for i in range(1,numcuts):
        for j in range(i):
            rect = tiles[i][j]
            assert(isinstance(rect,Collist))

    if DEBUG:
        # DEBUG print tiles
        for til in tiles:
            for t in til:
                print(f'{t}: {t.nnz()} nnz')

def optimize_tiles(tiles,n):
    ''' Optimize and Merge tiles.
    Parameters:
    n: dimension of matrix
    tiles (list of lists): matrix of tiles

    Returns:
    tiles_list (list): unordered, unstructured list of lists of tiles
                       each list corresponds to a synchronization level
    '''
    # merge diagonal elements
    # TODO: DAG scheduling of subblocks
    numcuts = len(tiles)

    # merge all tiles below each other
    for c in range(numcuts-1):
        for r in range(c+2,numcuts):
            a = tiles[c+1][c]
            b = tiles[r][c]
            dprint(f'merge into tile{c+1} {c} tile{r} {c}')
            dprint(f'merge {b} into {a}')
            a.merge(b)
            tiles[r][c] = None

    # Collist tiles: only synchronize as much as necessary
    #bp_next = n
    #for c in reversed(range(numcuts-1)):
    #    t = tiles[c+1][c]
    #    t.set_bp_next(bp_next)
    #    bp_next = t.rowa

    #dprint("\n Determining bp_next")
    #for c in range(numcuts-1):
    #    t = tiles[c+1][c]
    #    dprint(f'{t}: bp_next = {t.bp_next}\t bp reduction range = [{t.rowa}-{t.bp_next}]')
    #dprint("\n")

    # remove empty tiles
    for til in tiles:
        for i in range(len(til)):
            t = til[i]
            if t is None or t.nnz() == 0:
                dprint(f'Remove empty tile: {t}')
                til[i] = None

    # schedule tiles to synchronization steps:
    tile_list = []
    for i in range(numcuts):
        # add diagonal triangle first
        t = tiles[i][i]
        if t is not None and t.nnz() > 0:
            tile_list.append([t])
            tiles[i][i] = None
        # add entire rectangle below the diagonal triangle
        if i+1 < numcuts:
            t = tiles[i+1][i]
            if t is not None and t.nnz() > 0:
                tile_list.append([t])
                tiles[i+1][i] = None

    # assert we processed all tiles
    for tr in tiles:
        for t in tr:
            assert(t == None)

    # assign reduction range from [self.rowa to self.reduce_stop)
    stop = n
    for t in reversed(tile_list):
        t = t[0]
        if t.REDUCES:
            t.reduce_stop = stop
            stop = t.rowa
            dprint(f'Assigning reduction of bp[{t.rowa},{t.reduce_stop}) to {t}.')

    return tile_list

def schedule_to_workers(tile_list):
    ''' Scheduling Triangles and Rectangles onto processor cores.
    Parameters:
    tile_list (list(list)): A list of synchronization steps. Each step can contain multiple tiles in a list, to represent multiple parallelizable workloads.
    Returns:
    schedule_fe (list(tuple)): A list of synchronization steps. Each step is a tuple of WORKER elements. Each element defines the work to do for that specific processing core.
    schedule_bs (list(tuple)): Same but for the Backward Substitution.
    '''

    schedule_fe, schedule_bs = [], []
    for work in tile_list:
        if len(work) > 1:
            raise NotImplementedError("Multiple tiles, so inter-kernel workload balancing, is unimplemented")
        tile = work[0]
        dprint(f'Scheduling {tile}')
        if tile.REDUCES:
            tile.schedule_reduction()

        # distribute work while balancing load
        dist = tile.schedule()

        # add dummy synch steps if some kernels synch interanlly as well

        # FE
        for i in range(len(dist)):
            if dist[i].assigned_data() == 0:
                if not isinstance(dist[i],Collist):
                    dist[i] = Empty(0,0,0,0)
        schedule_fe.append(tuple(dist))
        maxsnum = max([d.snum_fe() for d in dist])
        for i in range(maxsnum):
            buffer_dist = []
            for h in range(HARTS):
                if dist[h].snum_fe() > i:
                    buffer_dist.append(SynchBuffer(0,0,0,0))
                else:
                    buffer_dist.append(Empty(0,0,0,0))
            schedule_fe.append(tuple(buffer_dist))

        # BS
        # in FE we want to keep emtpy collist items: they serve of value
        # in BS we do not need these.
        # purge dist from empty items:
        for i in range(len(dist)):
            if dist[i].assigned_data() == 0:
                dist[i] = Empty(0,0,0,0)
        maxsnum = max([d.snum_bs() for d in dist])
        for i in range(maxsnum):
            buffer_dist = []
            for h in range(HARTS):
                if dist[h].snum_fe() > i:
                    buffer_dist.append(SynchBuffer(0,0,0,0))
                else:
                    buffer_dist.append(Empty(0,0,0,0))
            schedule_bs.append(tuple(buffer_dist))
        # add in reverse order because we reverse later
        schedule_bs.append(tuple(dist))

    # reverse 
    schedule_bs.reverse()
    return schedule_fe,schedule_bs


def print_schedule(schedule,s_offset=0):
    #print('\n########## SCHEDULING ##########')
    for synch,step in enumerate(schedule):
        print(f'synch. step {synch+s_offset}:')
        for hart,work in enumerate(step):
            if work.assigned_data() == 0:
                print(f'  H{hart} {work}: ')
            else:
                print(f'  H{hart} {work}:\t {work.assigned_data()} assigned elements')
        print()


def codegenSolver(problem,schedule_fe,schedule_bs,bp_sync):
    synchsteps_fe = len(schedule_fe)
    synchsteps_bs = len(schedule_bs)
    direc = f'./build/{problem}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    callfile = f'{direc}/parspl.c'
    datafile = f'{direc}/scheduled_data.h'
    bprint(f'\nCode generation into {callfile} and {datafile}.')
    print(f"Synchronizing write access to bp at: {bp_sync}")

    # Gather funcalls and data
    funcalls_fe = [[] for i in range(HARTS)]
    funcalls_bs = [[] for i in range(HARTS)]
    codedata = {}
    for s,dist in enumerate(schedule_fe):
        assert(len(dist) == HARTS)
        for h,d in enumerate(dist):
            # call tiles codegen
            solve,dat = d.codegen(s,h)
            # process data
            for k,v in dat.items():
                if k in codedata:
                    dprint(f'Discarding {k} for s{s}h{h}: duplicate')
                else:
                    codedata[k] = v
            codedata.update(dat)
            # process function call
            funcalls_fe[h].append(solve)

    for s,dist in enumerate(schedule_bs):
        assert(len(dist) == HARTS)
        for h,d in enumerate(dist):
            # call tiles codegen
            solve,dat = d.codegen(s,h)
            # process data
            for k,v in dat.items():
                if k in codedata:
                    dprint(f'Discarding {k} for s{s}h{h}: duplicate')
                else:
                    codedata[k] = v
            codedata.update(dat)
            # process function call
            funcalls_bs[h].append(solve)

    with open(callfile,'w') as f:
        f.write('#include "runtime.h"\n')
        f.write('#include "kernel.h"\n')
        f.write('#include "workspace.h"\n')
        f.write('#include "scheduled_data.h"\n\n')

        # create call to lsolve
        f.write('void lsolve(int core_id){\n')
        f.write('\tswitch (core_id){\n')
        for h in range(HARTS):
            f.write(f'\t\tcase {h}:\n')
            for s,(fun,_) in zip(range(synchsteps_fe),funcalls_fe[h]):
                d = schedule_fe[s][h]
                synchs = [str(i) for i in range(s,s+d.snum_fe()+1)]
                if not isinstance(d,SynchBuffer):
                    f.write(f'\t\t\t// synch step {" ".join(synchs)}\n')
                    f.write(f'\t\t\t{fun};\n')
                    f.write(BARRIER)
                #else:
                #    f.write(f'\t\t\t{fun};\n')
            f.write(f'\t\t\tbreak;\n')
        f.write(f'\t\tdefault:\n')
        f.write(f'\t\t\t#ifdef PRINTF\n')
        f.write(f'\t\t\tprintf("Error: wrong core count configuration in code generation.\\n");\n')
        f.write(f'\t\t\t#endif\n')
        for s in range(synchsteps_fe):
            f.write(f'\t\t\t// synch step {s}\n')
            f.write(BARRIER)
        f.write(f'\t\t\tbreak;\n')
        f.write('\t}\n}\n\n')

        # synchronization count in FE
        s_offset_bs = synchsteps_fe
        # diagonal inverse multiplication
        s_offset_bs += 1


        # create call to ltsolve
        f.write('void ltsolve(int core_id){\n')
        f.write('\tswitch (core_id){\n')
        for h in range(HARTS):
            f.write(f'\t\tcase {h}:\n')
            for s,(_,fun) in zip(range(synchsteps_bs),funcalls_bs[h]):
                d = schedule_bs[s][h]
                synchs = [str(i+s_offset_bs) for i in range(s,s+d.snum_bs()+1)]
                if not isinstance(d,SynchBuffer):
                    f.write(f'\t\t\t// synch step {" ".join(synchs)}\n')
                    f.write(f'\t\t\t{fun};\n')
                    f.write(BARRIER)
                #else:
                #    f.write(f'\t\t\t{fun};\n')
            f.write(f'\t\t\tbreak;\n')
        f.write(f'\t\tdefault:\n')
        f.write(f'\t\t\t#ifdef PRINTF\n')
        f.write(f'\t\t\tprintf("Error: wrong core count configuration in code generation.\\n");\n')
        f.write(f'\t\t\t#endif\n')
        for s in range(synchsteps_bs):
            f.write(f'\t\t\t// synch step {s+s_offset_bs}\n')
            f.write(BARRIER)
        f.write(f'\t\t\tbreak;\n')
        f.write('\t}\n}\n\n')

        # create call to solve
        f.write('void solve(int core_id){\n')
        f.write('\tlsolve(core_id);\n')
        f.write('\tdiag_inv_mult(core_id);\n')
        f.write('\tltsolve(core_id);\n')
        f.write('}\n')

    # dump data fo file
    with open(datafile,'w') as f:
        #f.write('#include "runtime.h"\n\n')
        for k,v in codedata.items():
            if isinstance(v,list):
                v = list2array(v,k)
            if isinstance(v,np.ndarray):
                ndarrayToC(f,k,v)
            else:
                raise NotImplementedError(f'Unknown how to convert {type(v)} to code')


def writeWorkspaceToFile(problem,linsys,permutation=None,case='lsolve',debug=False):
    global Kdata
    global Ldata
    global x_gold
    direc = f'./build/{problem}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    workh = f'{direc}/workspace.h'
    workc = f'{direc}/workspace.c'
    goldenh = f'{direc}/golden.h'
    bprint(f'\nCreating workspace')
    print(f'Dumping to {workh} and {workc}.\nIncluding golden model for **{case}**')

    try:
        fh = open(workh,'w')
        fc = open(workc,'w')

        # includes and defines
        fc.write('#include "workspace.h"\n\n')
        fh.write('#include <stdint.h>\n')
        fh.write(f'#define {case.upper()}\n')
        fh.write(f'#define LINSYS_N ({linsys.n})\n\n')

        # create golden model: M @ x_golden = bp
        # Determine M matrix depending on the verification case
        if case == 'solve':
            # TODO: use K directly to avoid any conversion errors
            # but since we still have LDLT as input data
            # and do not do the decomposition ourself it is safer to do this instead
            #M = spa.csc_matrix((linsys.Kx,linsys.Ki,linsys.Kp),shape=(linsys.n,linsys.n))
            M = Kdata
        elif case == 'ltsolve':
            M = Ldata.transpose()
        elif case == 'lsolve':
            M = Ldata
        else:
            raise NotImplementedError(f'Case {case} unknown')

        if debug and linsys.n < 20:
            print('Matrix for golden model creation:')
            print(M.toarray())


        fc.write(f'// verification of {case}\n')
        # b
        b = M @ x_gold
        ndarrayToCH(fc,fh,'b',b)
        # golden
        ndarrayToCH(fc,fh,'XGOLD',x_gold,section='')
        ndarrayToCH(fc,fh,'XGOLD_INV',1/x_gold,section='')
        # Perm
        Dinv = 1/np.array(linsys.D)
        if permutation is not None:
            perm = np.array(permutation)
            ndarrayToCH(fc,fh,'Perm',perm)
            ndarrayToCH(fc,fh,'PermT',np.argsort(perm))
            # bp, bp_copy
            #wprint("TODO: do not use in production: bp preset")
            bp = np.empty(linsys.n)
            #bp = b[permutation]
            ndarrayToCH(fc,fh,'bp',bp)
            # Dinv
            #permT = np.argsort(perm)
            Dinv = Dinv[perm]
            ndarrayToCH(fc,fh,'Dinv',Dinv)
        else:
            # Dinv
            ndarrayToCH(fc,fh,'Dinv',Dinv)
        # solution vector x
        x = np.empty(linsys.n)
        ndarrayToCH(fc,fh,'x',x)
        bp_cp = np.zeros(linsys.n)
        ndarrayToCH(fc,fh,'bp_cp',bp_cp)
        # temporary space for intermediate results before reduction
        for h in range(HARTS):
            bp_tmp = np.zeros(linsys.n)
            ndarrayToCH(fc,fh,f'bp_tmp{h}',bp_tmp)
    finally:
        fh.close()
        fc.close()

def interactive_plot_schedule(L,schedule,cuts):
    (fig,ax,sq_dict) = L.plot(uselx=False)

    # legend
    elems = []
    for h,c in zip(range(HARTS),color_palette):
        patch = Patch(facecolor=c, edgecolor=None, label=f'Hart {h}')
        elems.append(patch)
    ax.legend(handles=elems, loc='upper right')

    # Fade color
    def fade_all(sq_dict):
        grey = (0.7,0.7,0.7)
        for r in sq_dict:
            for v in sq_dict[r].values():
                v.set_color(grey)
    # Color according to schedule, run loop interactively
    fade_all(sq_dict)
    plt.pause(0.001)
    patchlist = []
    prompt = escape.BOLD + f'Select Schedule from 0..{len(schedule)-1}: '
    prompt += escape.END
    while(True):
        try: # get user input
            selection = int(input(prompt))
            if selection >= len(schedule):
                raise ValueError()
        except EOFError as e:
            break
        except Exception as e:
            wprint(f'Exception: default to schedule 0')
            selection = 0
        tmp_schedule = [schedule[selection]]
        # clean up plot
        for patch in patchlist:
            patch.remove()
        patchlist = []
        plotobjs = []
        fade_all(sq_dict)
        # recolour
        for synch_num,work in zip([selection],tmp_schedule):
            for h,tile in enumerate(work):
                if tile is None:
                    pass
                else:
                    print(f' H{h} {tile}')
                    patches = tile.color_dict(sq_dict,color_palette[h])
                    patchlist.extend(patches)
                    rect = tile.show_on_plot(ax,number=synch_num)
                    plotobjs.extend(rect)
                #elif kernel is Kernel.DENSETRIAG:
                #    plist = data.color_dict(sq_dict,fillin=True)
                #    for box in plist:
                #        patchlist.append(ax.add_patch(box))
                #elif kernel is Kernel.SPARSETRIAG:
                #    # sparse diag is entirely done by Heart 0
                #    for (r,c) in data.index:
                #        sq_dict[r][c].set_color(color_palette[0])
        plt.pause(0.001)

def plot_schedule(L,schedule,cuts):
    (fig,ax,sq_dict) = L.plot(uselx=False)

    # legend
    elems = []
    for h,c in zip(range(HARTS),color_palette):
        patch = Patch(facecolor=c, edgecolor=None, label=f'Hart {h}')
        elems.append(patch)
    ax.legend(handles=elems, loc='upper right')

    # Color according to schedule
    patchlist = []
    for synch_num,work in enumerate(schedule):
        for h,tile in enumerate(work):
            if tile is None:
                pass
            else:
                patches = tile.color_dict(sq_dict,color_palette[h])
                for p in patches:
                    patchlist.append(ax.add_patch(p))
                tile.show_on_plot(ax,number=synch_num)
    plt.show()


def read_cuts(problem,wd):
    ''' Read new-line seperated list from file '''
    cutfile = f'{wd}/src/{problem}.cut'
    cuts = []
    with open(cutfile,'r') as f:
        for l in f.readlines():
            cuts.append(int(l))
    return cuts

def cut2lines(cuts):
    # convert cut array to lines (points in x,y)
    x = []; y = []
    n = cuts[-1]
    s = n
    for s2 in cuts[1:-1]:
        #x.extend([0.5,s2+0.5,s2+0.5,None])
        #y.extend([s2+0.5,s2+0.5,s+0.5,None])
        x.extend([-0.5,s2-0.5,s2-0.5,None])
        y.extend([s2-0.5,s2-0.5,n-0.5,None])
        s = s2
    return (x,y)

def plot_cuts(problem,ax,n,wd):
    # read in cuts array
    cuts = read_cuts(problem,wd)
    verify_cuts(cuts,n) # verify + sort
    print(f"redrawing cuts at: {cuts}")
    (x,y) = cut2lines(cuts)
    lines = ax.plot(x,y,color='r',linewidth=1.5)
    return lines

def live_cuts(problem,L,wd,uselx=True):
    ''' Live cut matrix visually.'''
    # cuts
    cutfile = f'src/{problem}.cut'
    print(escape.BOLD, f'User should edit cutfile {cutfile}.',escape.END)
    # matrix
    (fig,ax,sq_dict) = L.plot(diag=False,uselx=uselx)
    plt.pause(0.001)
    lines = []
    # update cuts from file livecuts
    while(True):
        # update the artist data
        for l in lines:
            l.remove()
        # read in cuts array
        lines = plot_cuts(problem,ax,L.n,wd)
        # redraw
        plt.pause(2)


def simd_blocking(problem,wp):
    raise NotImplementedError()


def row_col_occupation(L):
    row_occ = np.zeros(L.n,dtype=int)
    col_occ = np.zeros(L.n,dtype=int)
    Lp,Li = (L.Lp,L.Li)
    for col in range(L.n):
        for k in range(Lp[col],Lp[col+1]):
            row = Li[k]
            row_occ[row] += 1
            col_occ[col] += 1
    return (row_occ,col_occ)
    

def compute_level(L):
    '''
    Compute levels from level scheduling of the L matrix.
    
    Parameters:
    L (Csc): Lower triangular matrix in Csc format and zero diagonal.

    Returns:
    np.ndarray: Vector of levels.
    dict: Bining dictionary that containts how many columns are in each level (starting with level 0).
    '''
    Lp,Li = (L.Lp,L.Li)
    # compute levels
    level = np.zeros(L.n)
    for col in range(L.n):
        for k in range(Lp[col],Lp[col+1]):
            level[Li[k]] = max(level[Li[k]],1+level[col])
    # bin levels
    bins = {}
    for i in level:
        if i not in bins:
            bins[i] = 1
        else:
            bins[i] += 1
    return (level,bins)


def level2permutation(level,thr=None):
    '''
    Create reordering permutation matrix such that levels are sorted.
    
    Parameters:
    level (np.ndarray): Vector of levels.
    thr (int): Optional threshold. If provided only levels up to a threshold will be ordered.

    Returns:
    (perm,permT)
    perm (np.ndarray): Permutation vector representing the column permutation.
    permT (np.ndarray): Permutation vector representing the row permutation.
    '''
    n = len(level)
    if thr is None:
        thr = n+1

    #perm = np.argsort(level) # level[perm] will be a sorted array
    # argsort works but shuffels the currently allready quite good sorting.
    
    # Sort by level but keep the columns on the same level in the same order.
    # swap sort is O(n^2) but keeps original order.
    level = level.copy()
    perm = [x for x in range(n)]
    for i in range(1,n):
        for j in range(1,n-1):
            # check and swap larger value to higher indices
            if level[j] > level[j+1] and level[j+1] <= thr:
                level[j], level[j+1] = level[j+1], level[j]
                perm[j], perm[j+1] = perm[j+1], perm[j]
        pass

    # compute permT that is used to swap column indices.
    #  if perm is the column ordering than permT is the corresponding row ordering.
    permT = np.argsort(perm)
    return (perm,permT)

def permute_csc(L,perm,permT):
    '''
    Permute csc L matrix according to permutation vector.
    
    Parameters:
    L (Triag): Lower triangular matrix in Csc format and zero diagonal.
    perm (np.ndarray): Permutation vector representing the column permutation.
    permT (np.ndarray): Permutation vector representing the row permutation.

    Returns:
    Triag: Reordered L matrix.
    '''
    assert(L.n == len(perm))

    # empty data structure
    Lp,Li,Lx = (L.Lp,L.Li,L.Lx)
    PLp = np.empty(len(Lp),dtype=int)
    PLi = np.empty(len(Li),dtype=int)
    PLx = np.empty(len(Lx))

    # reorder columns and rows in one go
    # compute PLP^T
    progress = 0
    PLp[0] = 0
    for i,col in enumerate(perm):
        for elem in range(Lp[col],Lp[col+1]):
            PLi[progress] = permT[Li[elem]]
            PLx[progress] = Lx[elem]
            progress += 1
        PLp[i+1] = progress
    assert(progress == len(Li))
    PL = Triag(PLp,PLi,PLx,L.n,'Permuted L')
    return PL

def level_schedule(L):
    '''
    Compute and permute L according to level scheduling.
    
    Parameters:
    L (Csc): Lower triangular matrix in Csc format and zero diagonal.

    Returns:
    Csc: Reordered L matrix.
    perm (np.ndarray): Permutation vector.
    '''
    level,bins = compute_level(L)
    perm,permT = level2permutation(level)
    PL = permute_csc(L,perm,permT)
    return (PL,perm)


def incomplete_level_schedule(L,threshold, intra_level_reorder = False):
    '''
    Compute and permute L according to level scheduling.
    However only schedule the levels up until a threshold.
    
    Parameters:
    L (Csc): Lower triangular matrix in Csc format and zero diagonal.
    threshold (int): Threshold above which we do not permute anymore.
    intra_level_reorder (bool): If true sort inside a level the columns increasingly by occupation.

    Returns:
    Csc: Reordered L matrix.
    perm (np.ndarray): Permutation vector.
    '''
    level,bins = compute_level(L)
    perm,permT = level2permutation(level,threshold)
    perm = np.array(perm)
    PL = permute_csc(L,perm,permT)

    if intra_level_reorder:
        _, col_occ = row_col_occupation(PL)
        a = 0 # start
        for level in range(threshold):
            z = a + bins[level] # end of current level
            level_occ = col_occ[a:z]
            sorting = np.argsort(level_occ)
            perm[a:z] = perm[a:z][sorting]
            a = z
        permT = np.argsort(perm)
        PL = permute_csc(L,perm,permT)
    return (PL,perm)


def main(args):
    global DEBUG
    global Kdata
    global Ldata
    global x_gold
    if args.debug:
        DEBUG = True
        general.DEBUG = True
        plt.ion()

    filename = f'{args.wd}/src/{args.test}.json'
    cutfile = f'{args.wd}/src/{args.test}.cut'
    linsys = general.loadDotDict(filename)
    L = Triag(linsys.Lp,linsys.Li,linsys.Lx,linsys.n,f'L from {filename}')
    perm = None # permutation matrix

    ##### Give some metrics about input data:
    #if args.lsolve or args.ltsolve or args.solve:
    Ldata = spa.csc_matrix((L.Lx,L.Li,L.Lp),shape=(L.n,L.n)) + spa.eye(L.n)
    Kdata = spa.csc_matrix((linsys.Kx,linsys.Ki,linsys.Kp),shape=(L.n,L.n))
    if args.numerical_analysis:
        bprint('Statistics of Input Data')

        # condition number for numerical stability analysis
        for mat,nam in [(Ldata,"Ldata"),(Kdata,"Kdata")]:
            k = min(6,min(mat.shape)-1)
            #try:
            #    sv_max = spa.linalg.svds(mat, return_singular_vectors=False, k=k, which='LM')
            #    sv_min = spa.linalg.svds(mat, return_singular_vectors=False, k=k, which='SM')
            #    sv_min = min(sv_min)
            #    sv_max = max(sv_min)
            #    cond = sv_max/sv_min
            #    print(f'Conditional Number of {nam} = {cond:.1f}')
            #except Exception as e:
            cond = np.linalg.cond(mat.todense())
            print(f'Conditional Number of {nam} = {cond:.1f}')

    # randomly sample from 1e-{RANGE} to 1e{RANGE}
    if DEBUG:
        x_gold = np.ones(linsys.n)
    else:
        RANGE = 5
        exponent = np.random.random_sample(linsys.n)*(2*RANGE)-RANGE
        x_gold = np.exp(exponent)
        if args.numerical_analysis:
            print(f'Norm of x_gold = {np.linalg.norm(x_gold):.2e}')

    # information about performance of Lsolve, Ltsolve and solve
    if args.numerical_analysis:
        for mat,nam in reversed([(Ldata,"Ldata"),(Ldata.transpose(),"Ldata^T"),(Kdata,"Kdata")]):
            # generate data
            print(f'Evaluating spa.linalg.spsolve with {nam}:')
            b_rhs = mat@x_gold
            print(f'\tNorm of right hand side vector:\t |b| = {np.linalg.norm(b_rhs):.2e}')
            x_sol = spa.linalg.spsolve(mat,b_rhs)
            print(f'\tMaximum absolute error:\t\t max|g-x| = {np.max(np.abs(x_gold-x_sol)):.2e}')
            print(f'\tMaximum relative error:\t\t max|g-x|/|g|= {np.max(np.abs(x_gold-x_sol)/np.abs(x_gold)):.2e}')
            # compute errors
            relerr = np.linalg.norm(x_gold-x_sol)/np.linalg.norm(x_sol)
            print(f'\tRelative error norm:\t\t |gold-x|/|gold|= {relerr:.2e}')
            residual = b_rhs - mat@x_sol
            relerr = np.linalg.norm(residual)/np.linalg.norm(b_rhs)
            relres = 0
            print(f'\tRelative residual error norm:\t |b-Ax|/|b| = {relres:.2e}')
            print()


    #####  Do user requested exploration #####
    if args.gray_plot:
        args.plot = True

    if args.invert:
        # compute inverse of LDL.T = A
        Lmat= spa.csc_matrix((L.Lx,L.Li,L.Lp)) + spa.eye(L.n)
        Linv = spa.linalg.spsolve(Lmat,np.eye(L.n))
        thr = 1e-15
        mask = np.abs(Linv) < thr
        Linv[mask] = 0.0
        wprint('Masking Linv. Absolute values below {thr} are set to 0. {np.sum(mask)} are affected.')
        Linv = spa.csc_matrix(Linv) - spa.eye(L.n)
        Lp = Linv.indptr
        Li = Linv.indices
        Lx = Linv.data
        Linv = Triag(Lp,Li,Lx,L.n,name='L inverted')
        print(f'nnz(L) = {L.nnz()}, nnz(L^-1) = {Linv.nnz()}')
        wprint('Continuing with Linv as L.')
        wprint('Expect numerical artifacts!')
        L = Linv
        linsys.D = None
        linsys.Lp = Lp
        linsys.Li = Li
        linsys.Lx = Lx

    if args.level:
        bprint("\nComputing (dependency) levels of L matrix:")
        levels,bins = compute_level(linsys)
        # bin levels and print bins as well as there size.
        bprint(' Level\t Columns:')
        for i in range(len(bins)):
            print(f'level {i:>3}: {bins[i]:>3}\t',end='')
            if (i+1) % 5 == 0:
                print()
        print()

    if args.level_schedule or args.level_thr is not None:
        if args.level_thr is not None:
            bprint(f"\nLevel schedule only levels up to {args.level_thr}:")
            PL,perm = incomplete_level_schedule(linsys,args.level_thr, intra_level_reorder=args.intra_level_reorder)
        else:
            bprint("\nLevel schedule and permute L matrix:")
            PL,perm = level_schedule(linsys)
            dprint('  perm=\n',perm)
        print('Working with permuted L matrix from now on. Setting L to PL.')
        L = PL

    # Read Cutfile
    if not os.path.exists(cutfile) or args.cut:
        if not os.path.exists(cutfile):
            print(f'Cut File {cutfile} does not exist. Proposing cuts.')
        else:
            print(f'Cut File {cutfile} exists. Overwriting.')
            cuts = read_cuts(args.test,args.wd)
            bprint(f'\nOrriginally cutting at: ',end='')
            print(*cuts)
        # suggesting cuts:
        cuts = [0]
        if args.level_schedule or args.level_thr:
            levels,bins = compute_level(L)
            numlevels = args.level_thr if args.level_thr is not None else len(bins)
            for l in range(numlevels):
                cuts.append(cuts[-1] + bins[l])
        cuts.append(L.n)
        # printing and dumping cuts
        bprint(f'\nSuggesting to cut at: ',end='')
        print(*cuts)
        fd = open(cutfile,'w')
        for c in cuts:
            fd.write(f'{str(c)}\n')
        fd.close()

    if args.occupation:
        bprint('\nCalculate row and column occupation:')
        (row_occ,col_occ) = row_col_occupation(L)
        for vec,name in zip([row_occ,col_occ],['row','col']):
            bprint(f' {name}\t occupation:',end=None)
            t = 0
            for i in range(len(vec)):
                cnt = vec[i]
                if cnt != 0:
                    print(f'{name} {i:>3}: {cnt:>3}\t',end='')
                    t += 1
                if (t+1) % 10 == 0:
                    t = 0
                    print()
            print()

    if args.simd_blocking:
        simd_blocking(linsys)

    if args.live_cuts:
        live_cuts(args.test,L,args.wd,uselx=not args.gray_plot)

    if args.schedule or args.codegen:
        cuts = read_cuts(args.test,args.wd)
        bprint(f'\nCutting at: ',end='')
        print(*cuts)
        print("L matrix to cut & schedule:", L)
        tiles = tile_L(L,cuts) # structured tiles
        assign_kernel_to_tile(tiles)
        tile_list = optimize_tiles(tiles,linsys.n) #unstructure tiles
        bprint(f'\nScheduling to {len(tile_list)} tiling steps.')
        schedule_fe,schedule_bs = schedule_to_workers(tile_list)
        if args.schedule:
            print('\n## Forward Elimination ##')
            print_schedule(schedule_fe)
            s_offset = len(schedule_fe)
            print('\n## Dinv vector-vector scaling ##')
            print(f'synch. step {s_offset}:')
            print('\n## Backward Substitution ##')
            print_schedule(schedule_bs,s_offset=s_offset+1)
        if args.interactive_schedule:
            interactive_plot_schedule(L,schedule_fe,cuts)
        elif args.plot:
            plot_schedule(L,schedule_fe,cuts)
    elif args.plot:
       (fig,ax,sq_dict) = L.plot(uselx = not args.gray_plot)
       plot_cuts(args.test,ax,L.n,wd=args.wd)
       plt.show()

    if args.codegen:
        codegenSolver(args.test,schedule_fe,schedule_bs,cuts)
        case = 'solve'
        if args.lsolve:
            case = 'lsolve'
        elif args.ltsolve:
            case = 'ltsolve'
        elif args.solve:
            case = 'solve'

        # check that permutation is actually permuting anything
        if perm is None:
            perm = range(linsys.n)
        perm = list2array(list(perm),'perm',base=16) # column permutation
        permT =  np.argsort(perm) # row permutation
        # swap: the linear sytem solver assumes perm is the row permutation and
        #       permT the column one
        #perm,permT = permT,perm
        writeWorkspaceToFile(args.test,linsys,permutation=perm,case=case,debug=args.debug)

    # Dump files
    if args.dumpbuild:
        subprocess.run(f'{CAT_CMD} ./build/{args.test}/*',shell=True)

    if args.link:
        if not args.codegen:
            wprint('Linking without regenerating code')
        bprint('\nLinking generated code to virtual verification environment')
        wd = args.wd
        links = [
        f'ln -sf ../build/{args.test}/parspl.c {wd}/virtual/parspl.c',
        f'ln -sf ../build/{args.test}/scheduled_data.h {wd}/virtual/scheduled_data.h',
        f'ln -sf ../build/{args.test}/workspace.c {wd}/virtual/workspace.c',
        f'ln -sf ../build/{args.test}/workspace.h {wd}/virtual/workspace.h' ]
        for l in links:
            print(l)
            subprocess.run(l,shell=True,check=True)

    if DEBUG:
        breakpoint()

    return vars()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    '--test', '-t', nargs='?',
    default='_HPC_3x3_H2',
    help='Testcase to take data from.',
)
parser.add_argument('--simd_blocking','-s', action='store_true',
    help='Explore SIMD')
parser.add_argument('--live_cuts', '-l' ,action='store_true',
    help='Interactively determine life cuts. Edit file livecuts to define the cuts for candidate X.')
parser.add_argument('--cut', '-c' , action='store_true',
    help='Propose cuts to partition matrix into regions.')
parser.add_argument('--schedule', '-S' , action='store_true',
    help='Schedule cut appart matrix.')
parser.add_argument('--interactive_schedule', action='store_true',
    help='Schedule matrix interactively..')
parser.add_argument('--plot', '-p' , action='store_true',
    help='Color plot scheduled matrix.')
parser.add_argument('--modolo', '-m' , action='store_true',
    help='Color Matrix accodring to modolo schedule.')
parser.add_argument('--codegen', '-g' , action='store_true',
    help='Generate Code Structures from Schedule.')
parser.add_argument('--level', action='store_true',
    help='Compute level vector derived by level scheduling the L matrix. Show the levels and the amount of columns inside of each.')
parser.add_argument('--level_schedule', action='store_true',
    help='Level schedule and permute L matrix.')
parser.add_argument('--level_thr', type=int, default=None,
    help='Level threshold for partly level scheduling.')
parser.add_argument('--occupation', action='store_true',
    help='Compute row/column occupation, so number of elements.')
parser.add_argument('--intra_level_reorder', action='store_true',
    help='Permute, so sort increasingly, inside the levels by column occupation.')
parser.add_argument('--gray_plot', action='store_true',
    help='Plot cells in grey instead of color them by value.')
parser.add_argument('--debug', action='store_true',
    help='Debug print a lot of information. Use on small matrices.')
parser.add_argument('--invert', action='store_true',
    help='Invert L matrix')
parser.add_argument('--dumpbuild', action='store_true',
    help='Dump files in directory of problem to shell.')
parser.add_argument('--lsolve', action='store_true',
    help='Verify lsolve. Put golden model into workspace.')
parser.add_argument('--ltsolve', action='store_true',
    help='Verify ltsolve. Put golden model into workspace.')
parser.add_argument('--solve', action='store_true',
    help='Verify ldlsolve. Put golden model into workspace.')
parser.add_argument('--link', action='store_true',
    help='Link generated code to virtual verification environment.')
parser.add_argument('--wd', type=str, default='.',
    help='Working directory.')
parser.add_argument('--numerical_analysis', action='store_true',
    help='Working directory.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
