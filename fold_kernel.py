#!/usr/bin/env python3
import numpy as np
import math

def max_rel_err(x, gold, tol=1e-3,log=False):
    x = x.flatten()
    gold = gold.flatten()
    err = np.abs(x-gold)
    relerr = err/gold
    idx = np.argmax(relerr)
    m = relerr[idx]
    if m > tol:
        eprint(escape.RED,f'FUNCTIONAL ERROR: Max rel error = {m:.4e} at index {idx}',escape.END)
        eprint(f'Calculated {x[idx]} but want {gold[idx]}.')
    elif log:
        print(f'Max rel error = {m:.4e}')
        #eprint(f'x =    {x}\ngold = {gold}')
    return relerr


def Fsolve(F,n,m,b):
    s = len(b)
    x = np.zeros((s))
    x[0] = b[0]
    # go through row by row (of T) and compute dot-product
    # accessing a row of T in F is intricate
    for i in range(1,s):
        accu = 0
        print(f'index {i}:')
        base = i-1
        if (i-1 < n):
            for j in range(i):
                val = F[base+j*m] # vertical stream
                accu += b[j]*val  # stream from bp
                print(base,j,val)
        else:
            for j in range(n):
                val = F[base+j*m] # vertical stream
                accu += b[j]*val  # strem from bp
                print(base,j,val)
            overshoot = i-n
            print(f'calc overshoot {overshoot}')
            base = (overshoot)*m
            for j in range(overshoot):
                val = F[base+j] # horizontal stream
                accu += b[n+j]*val # continue streaming from bp
                print(base,j,val)
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
            print(f'column {i}: base {base} (vertical)')
            for j in range(s-i-1):
                val = F[base+j] # horizontal stream
                bval = b[i+j+1]
                accu += bval*val
                print(base,j,val,bval)
        else:
            base = (i-n)*(m+1) + m # precompute in an array! to complex.
            print(f'column {i}: base {base} (horizontal)')
            for j in range(m-i):
                val = F[base+j*m] # horizontal stream
                bval = b[i+j+1]
                accu += bval*val
                print(base,j,val,bval)
        x[i] = b[i] + accu
    return x


def to_Fold(mTinv):
    ''' Convert triangular matrix to folded negative inverse'''
    s = mTinv.shape[0]
    (n,m) = (math.ceil(s/2),s-1)
    print(f"Folding {s}x{s} to {n}x{m}")
    F = np.zeros((n,m))
    for c in range(s-1):
        for r in range(c+1,s):
            if r == c:
                pass # skip diagonal
            elif c < n:
                F[c][r-1] = mTinv[r][c]
            else:
                F[r-n][c-n] = mTinv[r][c]
    return F


def func_verify_Fsolve(s):
    # generate triangular matrix
    T = np.eye(s)
    i = 0
    for c in range(s):
        for r in range(c+1,s):
            T[r][c] = i
            i += 1
    print(T)
    Tinv = np.linalg.inv(T)

    # convert to folded
    F = to_Fold(Tinv) # invert and store as folded
    (n,m) = F.shape
    print(F)
    F = F.flatten()
    b = np.array([i/2+0.56 for i in range(s)])

    # FE
    gold = Tinv @ b
    x = Fsolve(F,n,m,b)
    print(f'Solving Forward Elimination.')
    relerr = max_rel_err(x,gold,log=True)
    # BS
    gold = Tinv.transpose() @ b
    x = Ftsolve(F,n,m,b)
    print(f'Solving Backward Substition.')
    relerr = max_rel_err(x,gold,log=True)

if __name__ == "__main__":
    func_verify_Fsolve(4)
