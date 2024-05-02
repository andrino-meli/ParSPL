#!/usr/bin/env python3
import numpy as np
import argparse
import math
import sys
from general import eprint
import data_converter
from data_converter import Kernel
from workspace_parser import WorkspaceParser
from csc import Csc, Triag, DenseTriag, Rect, MetaRowSched
from general import escape, NUM_CORES, HEARTS, max_rel_err, dprint
import scipy
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve_triangular

def Fsolve(F,n,m,b):
    s = len(b)
    x = np.zeros((s))
    x[0] = b[0]
    # go through row by row (of T) and compute dot-product
    # accessing a row of T in F is intricate
    for i in range(1,s):
        accu = 0
        dprint(f'index {i}:')
        base = i-1
        if (i-1 < n):
            for j in range(i):
                val = F[base+j*m] # vertical stream
                accu += b[j]*val  # stream from bp
                dprint(base,j,val)
        else:
            for j in range(n):
                val = F[base+j*m] # vertical stream
                accu += b[j]*val  # strem from bp
                dprint(base,j,val)
            overshoot = i-n
            dprint(f'calc overshoot {overshoot}')
            base = (overshoot)*m
            for j in range(overshoot):
                val = F[base+j] # horizontal stream
                accu += b[n+j]*val # continue streaming from bp
                dprint(base,j,val)
        x[i] = b[i] + accu
    return x


def Ftsolve(F,n,m,b):
    s = len(b)
    x = np.zeros((s))
    x[s-1] = b[s-1]
    # go through column by column (of T) and compute dot-product
    # accessing a column of T in F is easier than accessing a column
    for i in range(0,s-1):
        accu = 0
        if i < n:
            base = i*(m+1)
            dprint(f'column {i}: base {base} (vertical)')
            for j in range(s-i-1):
                val = F[base+j] # horizontal stream
                bval = b[i+j+1]
                accu += bval*val
                dprint(base,j,val,bval)
        else:
            base = (i-n)*(m+1) + m # precompute in an array! to complex.
            dprint(f'column {i}: base {base} (horizontal)')
            for j in range(m-i):
                val = F[base+j*m] # horizontal stream
                bval = b[i+j+1]
                accu += bval*val
                dprint(base,j,val,bval)
        x[i] = b[i] + accu
    return x


def Lsolve(codedata,bp):
    # TODO: think about synchro
    bwrite = bp.copy()
    bread = bp
    # compute
    steps = codedata['STEPS']
    bulk = codedata['bulk']
    kernel = codedata['kernel']
    for step in range(steps):
        for heart in HEARTS:
            k = kernel[heart][step]
            b = bulk[heart][step]
            logstr = f'S{step} H{heart} '+ k.name
            # determine writeback location
            #if k is Kernel.DENSETRIAG:
            #    bwrite = bpalt
            #else:
            #    bwrite = bp # solve inplace
            # solve row based kernels
            if k is Kernel.METAROW or k is Kernel.DENSETRIAG:
                if args.vverbose or args.verbose:
                    print(f'{logstr:<23}',b.strH(heart))
                Rx   = b.PRx[heart] # stream off floats
                Ri   = b.PRi[heart] # stream off col indexes
                slen = b.slen[heart] # length of streams
                meta = b.Pmeta[heart]
                ix = 0
                for row,length in meta: # iterate over rows
                    accu = bread[row]
                    for _ in range(length): # frep
                        if args.vverbose:
                            print(f'\tRx{Rx[ix]}, b[Ri[ix]]={bread[Ri[ix]]}')
                        if k is Kernel.METAROW:
                            accu -= Rx[ix]*bread[Ri[ix]]
                        elif k is Kernel.DENSETRIAG:
                            accu += Rx[ix]*bread[Ri[ix]]
                        ix += 1
                    bwrite[row] = accu
                    if args.vverbose:
                        print(f'\tUpdating bp[{row}] to {bwrite[row]}')
            # solve index based kernels
            elif k is Kernel.SPARSETRIAG:
                if args.vverbose or args.verbose:
                    print(f'{logstr:<23}',b)
                index = b.index
                data  = b.data # stream off data
                slen  = b.nnz # streaming length
                for val,ix in zip(data,range(slen)):
                    row,col = index[ix]
                    bwrite[row] -= bread[col]*val
                    if args.vverbose:
                        print(f'\tUpdating bp[{row}] to {bwrite[row]}')
            else:
                raise NotImplementedError(f'Kernel {k} is not implemented')
        # exchange bp data before synchronizatoin #TODO: implement this differently
        bread = bwrite.copy() # uff performance : )
        if args.vverbose or args.verbose:
            print(f'Synchronization after step {step}.')
    return bread

def Ltsolve(codedata,bp):
    # TODO: think about synchro
    bwrite = bp.copy()
    bread = bp
    # compute
    steps = codedata['STEPS']
    bulk = codedata['bulk']
    kernel = codedata['kernel']
    for step in reversed(range(steps)):
        print(f'step: {step}')
        for heart in HEARTS:
            k = kernel[heart][step]
            b = bulk[heart][step]
            logstr = f'S{step} H{heart} '+ k.name
            if k is Kernel.METAROW:
                if args.vverbose or args.verbose:
                    print(f'{logstr:<23}',b.strH(heart))
                Rx   = b.PRx[heart] # stream off floats
                Ri   = b.PRi[heart] # stream off col indexes
                slen = b.slen[heart] # length of streams
                meta = b.Pmeta[heart]
                ix = slen
                for row,length in reversed(meta): # iterate over rows
                    for tmp in range(length): # frep
                        breakpoint()
                        val = bread[ix]
                        bwrite[Ri[ix]] -= Rx[ix]*val
                        if args.vverbose:
                            print(f'\tRx{Rx[ix]}, b[Ri[{ix}]]-={Rx[ix]*val}')
                        ix -=1
                    if args.vverbose:
                        print(f'\tUpdating bp[{row}] to {bwrite[row]}')
            elif k is Kernel.DENSETRIAG:
                pass
            # solve index based kernels
            elif k is Kernel.SPARSETRIAG:
                if args.vverbose or args.verbose:
                    print(f'{logstr:<23}',b)
                index = b.index
                data  = b.data # stream off data
                slen  = b.nnz # streaming length
                for val,ix in zip(reversed(data),reversed(range(slen))):
                    col,row = index[ix]
                    bwrite[row] -= bread[col]*val
                    if args.vverbose:
                        print(f'\tUpdating bp[{row}] to {bwrite[row]}')
            else:
                raise NotImplementedError(f'Kernel {k} is not implemented')
        # exchange bp data before synchronizatoin #TODO: implement this differently
        bread = bwrite.copy() # uff performance : )
        if args.vverbose or args.verbose:
            print(f'Synchronization after step {step}.')
    return bread

def func_verify_Lsolve(L,codedata,dtype=np.float64,transpose=False):
    n = L.shape[0]
    bp = 1000*np.random.random_sample(n)-1
    x = bp.copy()
    #bp = np.ones(n)
    if transpose:
        Ldense = np.array(L.todense()).transpose() + np.eye(n)
        gold = scipy.linalg.solve_triangular(Ldense, bp, lower=False)
        x = Ltsolve(codedata,x)
    else:
        Ldense = np.array(L.todense()) + np.eye(n)
        #gold = spsolve_triangular(L, bp ,lower=True, unit_diagonal=True)
        gold = scipy.linalg.solve_triangular(Ldense, bp, lower=True)
        x = Lsolve(codedata,x)
    breakpoint()
    res = max_rel_err(x,gold)
    fun = np.abs((Ldense @ gold).flatten() - bp)
    gun = np.abs(Ldense @ x - bp).flatten()
    fi = np.argmax(fun)
    gi = np.argmax(gun)
    if (fun[fi] > 1e-3):
        eprint(escape.RED,f"Golden max err. {fun[fi]} after sol-back-mulp at index {fi}",escape.END)
    else:
        print(f"Golden max err. {fun[fi]} after sol-back-mulp at index {fi}")
    if (gun[gi] > 1e-3):
        eprint(escape.RED,f"SolveX max err. {gun[gi]} after sol-back-mulp at index {gi}",escape.END)
    else:
        print(f"SolveX max err. {gun[gi]} after sol-back-mulp at index {gi}")
    return res


def func_verify_Ltsolve(L,codedata,dtype=np.float64):
    return func_verify_Lsolve(L,codedata,dtype=dtype,transpose=True)


def func_verify_Fsolve(s):
    # generate triangular matrix
    T = np.eye(s)
    i = 0
    for c in range(s):
        for r in range(c+1,s):
            T[r][c] = i
            i += 1
    dprint(T)
    Tdense = DenseTriag(T)

    # convert to folded
    F = Tdense.to_Fold() # invert and store as folded
    (n,m) = F.shape
    dprint(F)
    F = F.flatten()
    b = np.array([i/2+0.56 for i in range(s)])
    dprint('b:',b)

    # FE
    gold = Tdense.Tinv @ b
    x = Fsolve(F,n,m,b)
    print(f'Solving Forward Elimination.')
    relerr = max_rel_err(x,gold,log=True)
    # BS
    gold = Tdense.Tinv.transpose() @ b
    x = Ftsolve(F,n,m,b)
    print(f'Solving Backward Substition.')
    relerr = max_rel_err(x,gold,log=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', '-p', nargs='?',
        default='_HPC_3x3_H2',
        help='Problem to be used.')
    parser.add_argument('--fsolve', '-f' , action='store_true',
        help='Verification of Fold solve.')
    parser.add_argument('--ldl', '-l' , action='store_true',
        help='Verification of LDLsolve')
    parser.add_argument('--lsolve', action='store_true',
        help='Verification of Lsolve')
    parser.add_argument('--ltsolve', action='store_true',
        help='Verification of Ltsolve')
    parser.add_argument('--debug', action='store_true',
        help='Use debug Matrix')
    parser.add_argument('--vverbose','-vv', action='store_true',
        help='Very Verbose logging.')
    parser.add_argument('--verbose','-v', action='store_true',
        help='Verbose logging.')

    args = parser.parse_args()

    if args.fsolve:
        func_verify_Fsolve(3)
        func_verify_Fsolve(6)
        func_verify_Fsolve(7)
        func_verify_Fsolve(8)
        func_verify_Fsolve(40) # ill-cond

    if args.lsolve or args.ltsolve:
        infile  = f'build/{args.test}/workspace_orig.c'
        outfile = f'build/{args.test}/workspace_meta.c'
        wp = WorkspaceParser(infile)
        L = wp.L
        codedata = data_converter.metaAll(L,args.test)
        L = csc_matrix((L.Lx,L.Li,L.Lp),shape=(L.n,L.n))
        if args.lsolve:
            res = func_verify_Lsolve(L,codedata)
        elif args.ltsolve:
            res = func_verify_Ltsolve(L,codedata)

    if args.debug:
        n = 8
        L = np.zeros((n,n))
        L[2][1] = 5; L[3][2] = 6
        L[4][0] = 11; L[5][0] = 12; L[6][0] = 13
        L[7][3] = 3; L[7][6] = 1
        L[4][3] = 20; L[5][3] = 22; L[6][3] = 22
        L[5][4] = 23; L[6][4] = 24
        L[6][5] = 25
        print('Using Debug Matrix to schedule')
        L = csc_matrix(L)
        L2 = Triag(L.indptr,L.indices,L.data,n,name='debug_L')
        codedata = data_converter.metaAll(L2,'debug_L')
        res = func_verify_Lsolve(L,codedata)
        breakpoint()




