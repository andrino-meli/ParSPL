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
from data_format import Collist, Tile, Diaginv, Empty
from general import escape, HARTS, eprint, wprint, bprint, NameSpace, dprint, color_palette, DEBUG, ndarrayToCH
import general

np.random.seed(0)
CAT_CMD = 'bat'
DEBUG = False

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
    bprint("Assigning Kernels to tiles:")
    numcuts = len(tiles)

    # decide on kernel for diagonal tiles
    for i in range(numcuts):
        triag = tiles[i][i]
        assert(isinstance(triag,Tile))
        if triag.nnz() == 0:
            tiles[i][i] = Empty(0,0,0,0)
        elif triag.density() > 0.8 or True: #TODO: make non_dense, non_empty diag kernels a thing
            print(f"DENSIFY: {triag}")
            tiles[i][i] = Diaginv(triag)
        elif triag.density() < 0.05:
            print(f"SPARSIFY: {triag}")
            raise NotImplementedError()
        else:
            eprint(f'Triag "{triag}" is neither sparse nor dense ({triag.density()*100:.1f}%). Inflating memory by inverting. Consider subcutting it.')
            tiles[i][i] = Diaginv(triag)
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

def optimize_tiles(tiles):
    ''' Optimize and Merge tiles.
    Parameters:
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
    return tile_list

def schedule_to_workers(tile_list):
    ''' Scheduling Triangles and Rectangles onto processor cores.
    Parameters:
    tile_list (list(list)): A list of synchronization steps. Each step can contain multiple tiles in a list, to represent multiple parallelizable workloads.
    Returns:
    schedule (list(tuple)): A list of synchronization steps. Each step is a tuple of WORKER elements. Each element defines the work to do for that specific processing core.
    '''

    schedule = []
    for synch_step, work in enumerate(tile_list):
        if len(work) > 1:
            raise NotImplementedError("Multiple tiles, so inter-kernel workload balancing, is unimplemented")
        tile = work[0]

        # distribute work while balancing load
        dist = tile.schedule()

        # purge dist from empty items:
        for i in range(len(dist)):
            if dist[i].assigned_data() == 0:
                dist[i] = Empty(0,0,0,0)
        schedule.append(tuple(dist))
    return schedule


def print_schedule(schedule):
    print('\n########## SCHEDULING ##########')
    for synch,step in enumerate(schedule):
        print(f'synch. step {synch}:')
        for hart,work in enumerate(step):
            if isinstance(work,Empty):
                print(f'  H{hart} empty')
            else:
                print(f'  H{hart} {work}:\t {work.assigned_data()} assigned elements')
        print()


def genCodeFromSched(schedule,bp_sync):
    bprint("Generating Code Structures from Scheduled Data.")
    print(f"Synchronizing write access to bp at: {bp_sync}")
    funcalls = [[] for i in range(HARTS)]
    codedata = {}
    for s,dist in enumerate(schedule):
        assert(len(dist) == HARTS)
        for h,d in enumerate(dist):
            solve,dat = d.codegen(s,h)
            funcalls[h].append(solve)
            for k,v in dat.items():
                assert(k[0:2] == f's{s}')
                if k in codedata:
                    dprint(f'Discarding {k} for s{s}h{h}: duplicate')
                else:
                    codedata[k] = v
            codedata.update(dat)
    return (funcalls,codedata)


def writeCodeToFile(problem,funcalls,codedata):
    SYNCHRONIZE = '__rt_barrier()'
    synchsteps = len(funcalls[0])
    direc = f'./build/{problem}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    callfile = f'{direc}/parspl.c'
    datafile = f'{direc}/scheduled_data.h'
    bprint(f'Dumping scheduled code to {callfile} and {datafile}.')

    # Create Call file for Lsolve
    with open(callfile,'w') as f:
        f.write('#include <stdio.h>\n')
        f.write('#include "runtime.h"\n')
        f.write('#include "kernel.h"\n')
        f.write('#include "scheduled_data.h"\n\n')
        f.write('void lsolve(int core_id){\n')
        f.write('\tswitch (core_id){\n')
        for h in range(HARTS):
            f.write(f'\t\tcase {h}:\n')
            for s,(fun,_) in zip(range(synchsteps),funcalls[h]):
                f.write(f'\t\t\t// synch step {s}\n')
                f.write(f'\t\t\t{fun};\n')
                f.write(f'\t\t\t{SYNCHRONIZE};\n')
            f.write(f'\t\t\tbreak;\n')
        f.write(f'\t\tdefault:\n')
        f.write(f'\t\t\tprintf("Error: wrong core count configuration in code generation.");\n')
        for s in range(synchsteps):
            f.write(f'\t\t\t// synch step {s}\n')
            f.write(f'\t\t\t{SYNCHRONIZE};\n')
        f.write(f'\t\t\tbreak;\n')
        f.write('\t}\n}')
    
    # Create Call file for Ltsolve
    # TODO!

    with open(datafile,'w') as f:
        #f.write('#include "runtime.h"\n\n')
        for k,v in codedata.items():
            if isinstance(v,list):
                v = general.list2array(v,k)
            if isinstance(v,np.ndarray):
                general.ndarrayToC(f,k,v)
            else:
                raise NotImplementedError(f'Unknown how to convert {type(v)} to code')


def writeWorkspaceToFile(problem,linsys,case='lsolve'):
    direc = f'./build/{problem}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    workh = f'{direc}/workspace.h'
    workc = f'{direc}/workspace.c'
    goldenh = f'{direc}/golden.h'
    bprint(f'Creating workspace {workh} and {workc}.')

    try:
        fh = open(workh,'w')
        fc = open(workc,'w')

        # includes and defines
        fc.write('#include "workspace.h"\n\n')
        fh.write(f'#define {case.upper()}\n')
        fh.write(f'#define LINSYS_N ({linsys.n})\n\n')

        # create golden model: M @ x_golden = bp
        ## randomly sample from 1e-{RANGE} to 1e{RANGE}
        RANGE = 5
        exponent = np.random.random_sample(linsys.n)*(2*RANGE)-RANGE
        x_gold = np.exp(exponent)
        # Determine M matrix depending on the verification case
        if case == 'ldlsolve':
            M = spa.csc_matrix((linsys.Kx,linsys.Ki,linsys.Kp),shape=(linsys.n,linsys.n))
            raise NotImplementedError()
        elif case == 'ltsolve':
            M = spa.csc_matrix((linsys.Lx,linsys.Li,linsys.Lp),shape=(linsys.n,linsys.n))
            M.transpose()
            M += spa.eye(linsys.n)
            raise NotImplementedError()
        elif case == 'lsolve':
            M = spa.csc_matrix((linsys.Lx,linsys.Li,linsys.Lp),shape=(linsys.n,linsys.n))
            M += spa.eye(linsys.n)
        # bp
        bp = M @ x_gold

        # bp_copy
        bp_cp = np.zeros(linsys.n)
        ndarrayToCH(fc,fh,'bp_cp',bp_cp)
        fc.write(f'// verification of {case}\n')
        ndarrayToCH(fc,fh,'bp',bp)
        ndarrayToCH(fc,fh,'XGOLD',x_gold)
        ndarrayToCH(fc,fh,'XGOLD_INV',1/x_gold)
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


def read_cuts(problem):
    ''' Read new-line seperated list from file '''
    cutfile = f'src/{problem}.cut'
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

def live_cuts(problem,L,uselx=True):
    ''' Live cut matrix visually.'''
    # cuts
    cutfile = f'src/{problem}.cut'
    print(escape.BOLD, f'User should edit cutfile {cutfile}.',escape.END)
    # matrix
    (fig,ax,sq_dict) = L.plot(diag=False,uselx=uselx)
    plt.pause(0.001)
    line = []
    # update cuts from file livecuts
    while(True):
        # read in cuts array
        cuts = read_cuts(problem)
        verify_cuts(cuts,L.n) # verify + sort
        print(f"redrawing cuts at: {cuts}")
        # update the artist data
        for l in line:
            l.remove()
        (x,y) = cut2lines(cuts)
        line = ax.plot(x,y,color='r',linewidth=1.5)
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
    perm (np.ndarray): Permutation vector.

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

if __name__ == '__main__':
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

    args = parser.parse_args()
    filename = f'./src/{args.test}.json'
    cutfile = f'./src/{args.test}.cut'
    linsys = NameSpace.load_json(filename)
    L = Triag(linsys.Lp,linsys.Li,linsys.Lx,linsys.n,f'L from {filename}')

    #L = Triag(list(range(10),Li,Lx,n,name'debug dummy data matrix'

    #####  Do user requested exploration #####
    if args.debug:
        DEBUG = True
        general.DEBUG = True

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
        bprint("Computing (dependency) levels of L matrix:")
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
            bprint(f"Level schedule only levels up to {args.level_thr}:")
            PL,perm = incomplete_level_schedule(linsys,args.level_thr, intra_level_reorder=args.intra_level_reorder)
        else:
            bprint("Level schedule and permute L matrix:")
            PL,perm = level_schedule(linsys)
            dprint('  perm=\n',perm)
        wprint('Setting L to PL temporarilly. Do not use this in production!')
        L = PL

    # Read Cutfile
    if not os.path.exists(cutfile) or args.cut:
        if not os.path.exists(cutfile):
            print(f'Cut File {cutfile} does not exist. Proposing cuts.')
        else:
            print(f'Cut File {cutfile} exists. Overwriting.')
            cuts = read_cuts(args.test)
            bprint(f'Orriginally cutting at: ',end='')
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
        bprint(f'Suggesting to cut at: ',end='')
        print(*cuts)
        fd = open(cutfile,'w')
        for c in cuts:
            fd.write(f'{str(c)}\n')
        fd.close()

    if args.occupation:
        bprint('Calculate row and column occupation:')
        (row_occ,col_occ) = row_col_occupation(wp.L)
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
        live_cuts(args.test,L,uselx=not args.gray_plot)

    if args.schedule or args.codegen:
        cuts = read_cuts(args.test)
        bprint(f'Cutting at: ',end='')
        print(*cuts)
        print("L matrix to cut & schedule:", L)
        tiles = tile_L(L,cuts) # structured tiles
        assign_kernel_to_tile(tiles)
        tile_list = optimize_tiles(tiles) #unstructure tiles
        bprint(f'Scheduling to {len(tile_list)} synchronization steps.')
        schedule = schedule_to_workers(tile_list)
        if args.schedule:
            print_schedule(schedule)
        if args.interactive_schedule:
            interactive_plot_schedule(L,schedule,cuts)
        elif args.plot:
            plot_schedule(L,schedule,cuts)
    elif args.plot:
       uselx = not args.gray_plot
       L.plot(uselx = uselx)
       plt.show()

    if args.codegen:
        codedata = genCodeFromSched(schedule,cuts)
        writeCodeToFile(args.test,*codedata)
        writeWorkspaceToFile(args.test,linsys)

    # Dump files
    if args.dumpbuild:
        bprint("Dumping files:\n\n")
        subprocess.run(f'{CAT_CMD} ./build/{args.test}/*',shell=True)

    if DEBUG:
        breakpoint()
