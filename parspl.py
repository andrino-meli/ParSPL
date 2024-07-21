#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import numpy as np
import argparse, os, json, sys, subprocess
import argcomplete
from scipy.sparse import csc_matrix
import scipy.sparse as spa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bfloat16 import bfloat16
from data_format import Csc, Triag
from data_format import Kernel, Collist, Tile, DiagInv, Empty, SynchBuffer, Mapping, CscTile
from general import escape, HARTS, eprint, wprint, bprint, DotDict, dprint
from general import color_palette, DEBUG, ndarrayToCH, ndarrayToC, list2array, list_flatten
import general

np.random.seed(0)
CAT_CMD = 'bat'
DEBUG = False

def workload_distribute_L(newLp,newLi,newLx,dtype_x=np.float64):
    LpStart = [] # Offset into Lx.
    LpLenTmp = [ list() for i in range(HARTS) ]
    LxNewTmp = [ list() for i in range(HARTS) ]
    LiNewTmp = [ list() for i in range(HARTS) ]
    for i in range(len(newLp)-2):
        tmp = int(newLp[i+1] - newLp[i])
        jobs = tmp // HARTS
        leftover = tmp % HARTS
        # compute how many jobs each core has to do and append the correspoding
        #  data to LxNew and LiNew.
        for core_id in range(HARTS):
            tmp = -1
            if (leftover != 0 and leftover > core_id):
                tmp = jobs+1
            else:
                tmp = jobs
            #LpStart.append( len(LxNew) if tmp > 0 else -1 )
            LpLenTmp[core_id].append(tmp)
            index = [t*(HARTS) + core_id for t in range(tmp)]
            halalu = [] # tmp for blocked Lx data
            for l in index:
                    halalu.append(newLx[l+newLp[i]])
            LxNewTmp[core_id].extend(halalu)
            tmparr = [newLi[l+newLp[i]] for l in index]
            LiNewTmp[core_id].extend(tmparr)
    LpLen = []; LxNew = []; LiNew = []; LpLenSum = []; LiNewR = []
    for i in range(HARTS):
        LpLen.extend(LpLenTmp[i])
        LpLenSum.append(sum(LpLenTmp[i]))
        LxNew.extend(LxNewTmp[i])
        LiNew.extend(LiNewTmp[i])
        LiNewTmp[i].reverse()
        LiNewR.extend(LiNewTmp[i])
    LpStart = [0]
    for i in range(HARTS):
        LpStart.append(LpLenSum[i]+LpStart[i])
    print(f'inflating Lp to LpLen increasing size from {len(newLp)} to {len(LpStart)+len(LpLen)}.')
    print(f'Average length in LpLen is {sum(LpLen)/len(LpLen)}')
    # convert to numpy arrays
    LpStart = list2array(LpStart,'LpStart',base=32)
    LpLenSum = list2array(LpLenSum,'LpLenSum',base=32)
    LpLen = list2array(LpLen,'LpLen',base=8)
    LiNew = list2array(LiNew,'LiNew',base=16)
    LiNewR = list2array(LiNewR,'LiNewR',base=16)
    LxNew = np.array(LxNew,dtype=dtype_x)

    return (LpStart,LpLen,LpLenSum,LxNew,LiNew,LiNewR)

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
            tiles[i][j] = Collist(cuts[i],cuts[i+1]-1,cuts[j],cuts[j+1]-1)
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

    def collist_is_mapping(cl):
        assert(isinstance(cl,Collist))
        rows = set() # set keeping track which rows are allready occupied
        for col,(Li,Lx) in cl.items():
            if len(Li) > 1: # check each column has only one element
                return False
            for el in Li:
                if el in rows:
                    return False
            # add all rows that are present in curent col to rows set
            rows.update(Li)
        return True           

    # decide on kernel for diagonal tiles
    for i in range(numcuts):
        triag = tiles[i][i]
        assert(isinstance(triag,Tile))
        if triag.nnz() == 0:
            tiles[i][i] = Empty(0,0,0,0)
        elif collist_is_mapping(triag):
            print(f"MAPPING: {triag}")
            raise NotImplementedError()
            tiles[i][i] = Mapping(triag)
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
    # only assign mappings to rows where all non-diag tiles are mappings as well:
    # this is necessary for functional correctness (at least currently)
    MIN_ROW_SPAN_TO_AMORTIZE_MAPPINGS = 5
    all_mappings = {}
    for i in range(1,numcuts):
        mappings = []
        for j in range(i):
            rect = tiles[i][j]
            assert(isinstance(rect,Collist))
            if rect.empty():
                continue
            is_mapping = collist_is_mapping(rect)
            is_worth_it = (rect.rowz-rect.rowa) > MIN_ROW_SPAN_TO_AMORTIZE_MAPPINGS
            if is_mapping and is_worth_it:
                mappings.append(j)
            else:
                if is_mapping and not is_worth_it:
                    print(f'Ignoring Mapping {rect} due to {MIN_ROW_SPAN_TO_AMORTIZE_MAPPINGS=}.')
                # revert mappings list
                if len(mappings) != 0:
                    bprint(f'Ignoring Mapping in {mappings} due to {rect}.')
                mappings = []
                break
        for j in mappings:
            rect = tiles[i][j]
            print(f'MAPPING: {rect}')
            tiles[i][j] = Mapping(rect)

    if DEBUG:
        # DEBUG print tiles
        for til in tiles:
            for t in til:
                print(f'{t}: {t.nnz()} nnz')


def optimize_tiles(tiles,n,args):
    '''
    Optimize and Merge tiles.
    Do dependency tree based scheduling of tiles.
    Remove empty dependencies.
    Merge independent tiles on same optimization level.

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


    # merge all Collist below each other
    #  this basically ensures that collists are getting merged and scheduled ASAP
    #  when they are beneave each other
    #  otherwise we want them scheduled ALAP
    # remove emtpy tiles
    for c in range(numcuts-1):
        tilecol = []
        for r in range(c+1,numcuts):
            t = tiles[r][c]
            if t.empty():
                tiles[r][c] = None
                continue
            if isinstance(t,Collist):
                tilecol.append((t,r,c))
        if len(tilecol) == 0:
            continue
        for t,r,c in tilecol[1:]:
            dprint(f'merge {t} into {tilecol[0][0]}')
            tilecol[0][0].merge(t)
            tiles[r][c] = None

    # tighten bounds on collist tiles
    # -> reveals acctual dependencies
    for til in tiles:
        for t in til:
            fun = getattr(t, "tighten_bound", None)
            if callable(fun):
                t.tighten_bound()

    # create tile_list by extracting tiles in order
    tile_list = []
    for til in tiles:
        for t in til:
            if t is not None and t.nnz() > 0:
                tile_list.append([t])

    # optimize using As Late As Possible Scheduling
    if args.alap:
        tile_list = list_flatten(tile_list)
        # create dependency tree
        bprint('Building and optimizing dependency tree')
        dprint('==== Tiles ====')
        for t in tile_list:
            dprint(t)
        class Node:
            def __init__(self,tile):
                self.tile = tile
                self.child = set()
                self.parent = set()
                self.level = 0
            def add_dependency(self,other):
                if self is other:
                    return
                if self.is_child_of(other):
                    self.parent.add(other)
                if self.is_parent_of(other):
                    self.child.add(other)
            def is_parent_of(self,other):
                if self.tile.rowa > other.tile.colz:
                    return False
                if self.tile.rowz < other.tile.cola:
                    return False
                return True
            def is_child_of(self,other):
                return other.is_parent_of(self)
            def __repr__(self):
                ch = ''; ph = ''
                for c in self.child:
                    ch += str(c.tile) + ', '
                for p in self.parent:
                    ph += str(p.tile) + ', '
                return f'Node{self.tile} {self.level} -> child:[{ch}] parent:[{ph}]'
                #return f'Node{self.tile} {self.level} '
        tree = [Node(t) for t in tile_list]
        # build dependency tree
        for s in tree:
            for o in tree:
                s.add_dependency(o)
        dprint('==== Tree ====')
        for t in tree:
            dprint(t)

        def reset_level(tree):
            for t in tree:
                t.level = 0

        # show levels of DAG tree
        def asap_level(tree):
            def ASAP_level(node):
                maxlevel = node.level
                for c in node.child:
                    c.level = max(node.level+1,c.level)
                    maxlevel = max(maxlevel,ASAP_level(c))
                return maxlevel
            reset_level(tree)
            maxlevel = 0
            for root in tree:
                if len(root.parent) == 0:
                    newmax = ASAP_level(root)
                    maxlevel = max(maxlevel,newmax)
            level_list = [[] for i in range(maxlevel+1)]
            for node in tree:
                level_list[node.level].append(node)
            return level_list

        def alap_level(tree):
            def ALAP_level(node):
                minlevel = node.level
                for p in node.parent:
                    p.level = min(node.level-1,p.level)
                    minlevel = min(minlevel,ALAP_level(p))
                return minlevel
            reset_level(tree)
            minlevel = 0
            for leave in tree:
                if len(leave.child) == 0:
                    minlevel = min(minlevel, ALAP_level(leave))
            level_list = [[] for i in range(-minlevel+1)]
            for node in tree:
                level_list[-minlevel+node.level].append(node)
            return level_list

        def optimize_level(levels):
            for i,l in enumerate(levels):
                cl = None
                tmp = []
                for node in l:
                    if isinstance(node.tile,Collist):
                        if cl is None:
                            cl = node
                            continue
                        else: # merge
                            dprint(f'Merging on same DAG level: \n\t\t{cl}\n\t\t {node}')
                            cl.child.update(node.child)
                            cl.parent.update(node.parent)
                            cl.tile.merge(node.tile)
                            continue
                    tmp.append(node)
                if cl is not None:
                    tmp.append(cl)
                levels[i] = tmp
            return levels

        def print_levels(levels):
            for i,l in enumerate(levels):
                print(f'level {i:2}:')
                for j,el in enumerate(l):
                    print(f'{" "*5}| ',el)

        if DEBUG:
            print('\n\n ==== ASAP DAG tree levels ====')
            levels = asap_level(tree)
            assert len(tile_list) == len(list_flatten(levels))
            print_levels(levels)

            print('\n\n ==== ALAP DAG tree levels ====')
            levels = alap_level(tree)
            assert len(tile_list) == len(list_flatten(levels))
            print_levels(levels)

        print('\n\n ==== optimized DAG tree levels ====')
        levels = alap_level(tree)
        assert len(tile_list) == len(list_flatten(levels))
        levels = optimize_level(levels)
        print_levels(levels)
        tile_list = []
        for l in levels:
            for node in l:
                tile_list.append([node.tile])

    # determining firstbp/lastbp to remove non-used reduction range
    firstbp = n
    lastbp = 0 #exclusive
    for tl in tile_list:
        for t in tl:
            if t.REDUCES:
                firstbp = min(firstbp,t.rowa)
                lastbp = max(lastbp,t.rowz+1)

    # assign reduction range from [self.rowa to self.reduce_stop)
    stop = lastbp
    for t in reversed(tile_list):
        assert len(t) == 1
        t = t[0]
        if t.REDUCES:
            t.reduce_stop = stop
            if stop <= t.rowa:
                wprint(f'{t} should be mergable into another Collist.')
            stop = min(t.rowa,stop)

    return tile_list,firstbp,lastbp

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


def codegenSolver(args,schedule_fe,schedule_bs,bp_sync):
    synchsteps_fe = len(schedule_fe)
    synchsteps_bs = len(schedule_bs)
    direc = f'{args.wd}/build/{args.test}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    datafile = f'{direc}/scheduled_data.h'
    bprint(f'\nCode generation into {datafile}.')
    print(f"Synchronizing write access to bp at: {bp_sync}")

    # define space for data that defines what kernel to call, what arguments to pass and
    #  what data to include
    enum = [[] for i in range(HARTS+1)] # defines function call to kernel
    argstruct = [[] for i in range(HARTS+1)] # defines passed argument in function call
    codedata = {} # defines static data

    def generate_enum_list(for_fe=True):
        for s,dist in enumerate(schedule_fe if for_fe else schedule_bs):
            assert(len(dist) == HARTS)
            for h,d in enumerate(dist):
                # call tiles codegen
                (kfe,kbs),(args_fe,args_bs),dat = d.codegen(s,h)
                if for_fe:
                    kernel,args = kfe, args_fe
                else:
                    kernel,args = kbs, args_bs
                if kernel is None:
                    continue
                # process data
                codedata.update(dat)
                # process function call
                enum[h].append(kernel.name)
                # process function arguments
                if args is not None:
                    argstruct[h].append('&'+args)
            # all other harts simply synchronize
            enum[-1].append(Kernel.SYNCH.name)

    # generate data according to FE schedule
    if args.case == 'solve' or args.case == 'lsolve':
        print('generating schedule for FE')
        generate_enum_list(for_fe=True)
    # process diag_inv_mult kernel
    if args.case == 'solve':
        print('generating schedule for inverse diagonal multiplication')
        for h in range(HARTS):
            enum[h].append(Kernel.DIAG_INV_MULT.name)
        enum[-1].append(Kernel.SYNCH.name)
    # process BS
    if args.case == 'solve' or args.case == 'ltsolve':
        print('generating schedule for BS')
        generate_enum_list(for_fe=False)


    # dump data fo file
    with open(datafile,'w') as f:
        # enumerate definition for Kernel
        enumdef = 'enum Kernel {'
        for k in (Kernel):
            enumdef += f'{k.name} = {k.value}, '
        enumdef += '};\n\n'
        f.write(enumdef)

        # dump static data
        for k,v in codedata.items():
            if isinstance(v,list):
                v = list2array(v,k)
                ndarrayToC(f,k,v,const=True)
            elif isinstance(v,np.ndarray):
                ndarrayToC(f,k,v,const=True)
            elif isinstance(v,str):
                v = v.split('=')
                assert len(v) == 2
                attr = f'__attribute__((aligned(8),section(".tcdm")))'
                f.write(f'const {v[0]} {attr} ={v[1]}\n')
            else:
                raise NotImplementedError(f'Unknown how to convert {type(v)} to code')

        # merge argument data
        args_coff = []
        argstruct_joined = []
        for v in argstruct:
            args_coff.append(len(argstruct_joined))
            argstruct_joined.extend(v)
        args_coff.append(len(argstruct_joined))
        # dump argument data
        name = 'argstruct_coreoffset'
        # TODO: select base for argstruct_joined: currently is uint8_t
        ndarrayToC(f,name,list2array(args_coff,name),const=True)
        attr = f'__attribute__((aligned(4),section(".tcdm")))'
        f.write(f'void * const argstruct_joined [] {attr} = ' + '{\n')
        for h,l in enumerate(argstruct):
            f.write(f'// HART {h}\n')
            for s in l:
                f.write(f'(void *) {s},\n')
        f.write('};\n\n')

        # merge enum data
        enum_coff = []
        enum_joined = []
        for v in enum:
            enum_coff.append(len(enum_joined))
            enum_joined.extend(v)
        enum_coff.append(len(enum_joined))
        # dump enum data
        name = 'enum_coreoffset'
        # TODO: select base for enum_coreoffset: currently is uint8_t
        ndarrayToC(f,name,list2array(enum_coff,name),const=True)
        attr = f'__attribute__((aligned(4),section(".tcdm")))'
        f.write(f'const enum Kernel enum_joined [] {attr} = ' + '{\n')
        for h,l in enumerate(enum):
            f.write(f'// HART {h}\n')
            for s in l:
                f.write(f'{s},\n')
        f.write('};\n\n')


def writeWorkspaceToFile(args,linsys,firstbp=0,lastbp=None,permutation=None):
    if lastbp is None:
        lastbp = linsys.n
    global Kdata
    global Ldata
    global x_gold
    direc = f'{args.wd}/build/{args.test}'
    if not os.path.exists(direc):
        os.makedirs(direc)
    workh = f'{direc}/workspace.h'
    workc = f'{direc}/workspace.c'
    goldenh = f'{direc}/olden.h'
    bprint(f'\nCreating workspace')
    case = 'solve'
    if args.lsolve:
        case = 'lsolve'
    elif args.ltsolve:
        case = 'ltsolve'
    print(f'Dumping to {workh} and {workc}.\nIncluding golden model for **{args.case}**')

    try:
        fh = open(workh,'w')
        fc = open(workc,'w')

        # includes and defines
        if args.sssr:
            fh.write(f'#define SSSR \n')
        if args.parspl:
            fh.write(f'#define PARSPL \n')
            fh.write(f'#define PERMUTATE \n')
        elif args.solve_csc:
            fh.write(f'#define SOLVE_CSC \n')
            permutation = None
        elif args.psolve_csc:
            fh.write(f'#define PSOLVE_CSC \n')
            permutation = None
        elif args.sssr_psolve_csc:
            fh.write(f'#define SSSR \n')
            fh.write(f'#define SSSR_PSOLVE_CSC \n')
            permutation = None
        fc.write('#include "workspace.h"\n')
        fh.write('#include <stdint.h>\n')
        fh.write(f'#define {args.case.upper()}\n')
        fh.write(f'#define LINSYS_N ({linsys.n})\n')
        fh.write(f'#define FIRST_BP ({firstbp})\n')
        fh.write(f'#define LAST_BP ({lastbp})\n')
         

        # create golden model: M @ x_golden = bp
        # Determine M matrix depending on the verification case
        if args.ltsolve:
            M = Ldata.transpose()
        elif args.lsolve:
            M = Ldata
        else:
            # TODO: use K directly to avoid any conversion errors
            # but since we still have LDLT as input data
            # and do not do the decomposition ourself it is safer to do this instead
            #M = spa.csc_matrix((linsys.Kx,linsys.Ki,linsys.Kp),shape=(linsys.n,linsys.n))
            M = Kdata

        if args.debug and linsys.n < 20:
            print('Matrix for golden model creation:')
            print(M.toarray())



        fc.write(f'// verification of {args.case}\n')
        # b
        b = M @ x_gold
        ndarrayToCH(fc,fh,'b',b)
        # golden
        ndarrayToCH(fc,fh,'XGOLD',x_gold,section='')
        ndarrayToCH(fc,fh,'XGOLD_INV',1/x_gold,section='')

        # CSC format data
        if args.sssr_psolve_csc:
            ddll = workload_distribute_L(linsys.Lp,linsys.Li,linsys.Lx)
            (LpStart,LpLen,LpLenSum,LxNew,LiNew,LiNewR) = ddll 
            ndarrayToCH(fc,fh,'LpStart',LpStart)
            ndarrayToCH(fc,fh,'LpLen',LpLen)
            ndarrayToCH(fc,fh,'LpLenSum',LpLenSum)
            ndarrayToCH(fc,fh,'LxNew',LxNew)
            ndarrayToCH(fc,fh,'LiNew',LiNew)
            ndarrayToCH(fc,fh,'LiNewR',LiNewR)
        elif args.solve_csc or args.psolve_csc:
            Lp = list2array(linsys.Lp,'Lp',base=32)
            Li = list2array(linsys.Li,'Li',base=16)
            Lx = np.array(linsys.Lx,dtype=np.float64)
            ndarrayToCH(fc,fh,'Lp',Lp)
            ndarrayToCH(fc,fh,'Li',Li)
            ndarrayToCH(fc,fh,'Lx',Lx)

        Dinv = 1/np.array(linsys.D)
        # Perm
        if permutation is not None:
            perm = np.array(permutation)
            if args.sssr:
                def lenstart_schedule(total,name,lenmo=True):
                    len_perm = []
                    start_perm = []
                    for h in range(HARTS):
                        l = (HARTS - 1 + total - h) // HARTS
                        if lenmo: #if length minus one
                            start_perm.append(sum(len_perm)+h)
                            len_perm.append(l-1) # frep/sssr length uses length - 1
                        else:
                            start_perm.append(sum(len_perm))
                            len_perm.append(l)
                    ndarrayToCH(fc,fh,f'start_{name}',list2array(start_perm,f'start_{name}',base=32))
                    ndarrayToCH(fc,fh,f'len_{name}',list2array(len_perm,f'len_{name}',base=32))
                lenstart_schedule(linsys.n,'perm')
            ndarrayToCH(fc,fh,'Perm',perm)
            ndarrayToCH(fc,fh,'PermT',np.argsort(perm))
            # bp, bp_copy
            RANGE = 5
            exponent = np.random.random_sample(linsys.n)*(2*RANGE)-RANGE
            bp = np.exp(exponent)
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
        # TODO: setting to random is only used for verification / testing
        RANGE = 5
        exponent = np.random.random_sample(linsys.n)*(2*RANGE)-RANGE
        x = np.exp(exponent)
        ndarrayToCH(fc,fh,'x',x)
        exponent = np.random.random_sample(linsys.n)*(2*RANGE)-RANGE
        bp_cp = np.exp(exponent)
        ndarrayToCH(fc,fh,'bp_cp',bp_cp)
        # temporary space for intermediate results before reduction
        bp_tmp = np.zeros(HARTS*(lastbp-firstbp))
        ndarrayToCH(fc,fh,f'bp_tmp',bp_tmp)
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

    args.parspl = True # default
    if args.solve_csc or args.psolve_csc or args.sssr_psolve_csc:
        args.parspl = False

    case = 'solve'
    if args.lsolve:
        case = 'lsolve'
    elif args.ltsolve:
        case = 'ltsolve'
    elif args.solve:
        case = 'solve'
    if case != 'solve' and not args.parspl:
        raise NotImplementedError('All other solving methods other than parspl do not support running FE or BS exclusively.')
    args.case = case

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
        if args.parspl:
            cuts = read_cuts(args.test,args.wd)
            bprint(f'\nCutting at: ',end='')
            print(*cuts)
            print("L matrix to cut & schedule:", L)
            tiles = tile_L(L,cuts) # structured tiles
            assign_kernel_to_tile(tiles)
            tile_list,firstbp,lastbp = optimize_tiles(tiles,linsys.n,args) #unstructure tiles
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
        elif args.solve_csc:
            cuts = None
            schedule_fe = [[Empty(0,0,0,0) for h in range(HARTS)]]
            schedule_bs = [[Empty(0,0,0,0) for h in range(HARTS)]]
            schedule_fe[0][0] = CscTile(linsys)
            schedule_bs[0][0] = CscTile(linsys)
        elif args.psolve_csc or args.sssr_psolve_csc:
            cuts = None
            schedule_fe = [[CscTile(linsys) for h in range(HARTS)]]
            schedule_bs = [[CscTile(linsys) for h in range(HARTS)]]
    elif args.plot:
       (fig,ax,sq_dict) = L.plot(uselx = not args.gray_plot)
       plot_cuts(args.test,ax,L.n,wd=args.wd)
       plt.show()

    if args.codegen and args.parspl:
        codegenSolver(args,schedule_fe,schedule_bs,cuts)

        # check that permutation is actually permuting anything
        if perm is None:
            perm = range(linsys.n)
        perm = list2array(list(perm),'perm',base=16) # column permutation
        permT =  np.argsort(perm) # row permutation
        # swap: the linear sytem solver assumes perm is the row permutation and
        #       permT the column one
        #perm,permT = permT,perm
        print(f'Using firstbp {firstbp}, lastbp {lastbp} for code generation of bp_tmp_h')
        writeWorkspaceToFile(args,linsys,firstbp=firstbp,lastbp=lastbp,permutation=perm)

    if args.solve_csc or args.psolve_csc or args.sssr_psolve_csc:
        writeWorkspaceToFile(args,linsys)

    # Dump files
    if args.dumpbuild:
        subprocess.run(f'{CAT_CMD} {args.wd}/build/{args.test}/*',shell=True)

    if args.link:
        if not args.codegen:
            wprint('Linking without regenerating code')
        bprint('\nLinking generated code to virtual verification environment')
        wd = args.wd
        links = [
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
parser.add_argument('--alap', action='store_true',
    help='As Late As possible scheduling in optimization.')
parser.add_argument('--sssr', action='store_true',
    help='Use Sparse Streaming Semantic Register hardware accelerations.')
parser.add_argument('--solve_csc', action='store_true',
    help='By default the generated code is using the ParSPL methodology. Specify this to generate code for the single core FE & BS using the common CSC matrix representation.')
parser.add_argument('--psolve_csc', action='store_true',
    help='Specify this to generate code for the multi-core FE & BS using the common CSC matrix representation.')
parser.add_argument('--sssr_psolve_csc', action='store_true',
    help='Specify this to generate code for the multi-core FE & BS using a distributed version of CSC optimized for SSSRs.')

if __name__ == '__main__':
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(args)
