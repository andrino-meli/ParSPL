#!/usr/bin/env python3
import numpy as np
import argparse
import os
from bfloat16 import bfloat16
from workspace_parser import WorkspaceParser
from scipy import sparse as sp
from scipy.linalg import ldl
from csc import Csc#, Triag, DenseTriag, SparseTriag, Rect, MetaRowSched
#from csc import color_palette
#import matplotlib.pyplot as plt
#from matplotlib.patches import Patch
from general import escape, NUM_CORES, HEARTS, eprint, wprint

UNKNOWN = -1;
USED = 1;
UNUSED = 0

# Convert LU decomposition into LDL
#  by https://services.math.duke.edu/~jdr/2021f-218/materials/week11.pdf
#  However does only work for Pos semidef matrices!
pass

# Compute the elimination tree for a quasidefinite matrix in CSC form
# code modified from: https://github.com/osqp/qdldl/blob/master/src/qdldl.c
#QDLDL_int QDLDL_etree(const QDLDL_int  n, const QDLDL_int* Ap, const QDLDL_int* Ai,
#                      QDLDL_int* work, QDLDL_int* Lnz, QDLDL_int* etree){
#  QDLDL_int i,j,p;
#  for(i = 0; i < n; i++){
#  // zero out Lnz and work.  Set all etree values to unknown
#    work[i]  = 0;
#    Lnz[i]   = 0;
#    etree[i] = QDLDL_UNKNOWN;
#
#    //Abort if A doesn't have at least one entry
#    //one entry in every column
#    if(Ap[i] == Ap[i+1]){
#      return -1;
#    }
#
#  }
#
#  for(j = 0; j < n; j++){
#    work[j] = j;
#    for(p = Ap[j]; p < Ap[j+1]; p++){
#      i = Ai[p];
#      if(i > j){return -1;}; //abort if entries on lower triangle
#      while(work[i] != j){
#        if(etree[i] == QDLDL_UNKNOWN){
#          etree[i] = j;
#        }
#        Lnz[i]++;         //nonzeros in this column
#        work[i] = j;
#        i = etree[i];
#      }
#    }
#  }


def compute_etree(n,Ap,Ai)
    # check input
    assert(len(Ap) = n+1)
    for i in range(n):
        assert(Ap[i+1] > Ap[1])

    # allocate space
    etree = UNKNOWN*np.ones(n)
    work = np.zeros(n)
    etree = np.zeros(n)

    # iterate through rows

    return etree


def sp_etree(K)
    K = sp.csc_matrix(K)
    n,m = L.shape
    assert(n == m)
    breakpoint()
    compute_etree(n,Kp,Ki)
#
#  QDLDL_int sumLnz;
#  //compute the total nonzeros in L.  This much
#  //space is required to store Li and Lx.  Return
#  //error code -2 if the nonzero count will overflow
#  //its unteger type.
#  sumLnz  = 0;
#  for(i = 0; i < n; i++){
#    if(sumLnz > QDLDL_INT_MAX - Lnz[i]){
#      sumLnz = -2;
#      break;
#    }
#    else{
#      sumLnz += Lnz[i];
#    }
#  }
#
#  return sumLnz;
#}


#
#
#
#QDLDL_int QDLDL_factor(const QDLDL_int    n,
#                  const QDLDL_int*   Ap,
#                  const QDLDL_int*   Ai,
#                  const QDLDL_float* Ax,
#                  QDLDL_int*   Lp,
#                  QDLDL_int*   Li,
#                  QDLDL_float* Lx,
#                  QDLDL_float* D,
#                  QDLDL_float* Dinv,
#                  const QDLDL_int* Lnz,
#                  const QDLDL_int* etree,
#                  QDLDL_bool*  bwork,
#                  QDLDL_int*   iwork,
#                  QDLDL_float* fwork){
#
#  QDLDL_int i,j,k,nnzY, bidx, cidx, nextIdx, nnzE, tmpIdx;
#  QDLDL_int *yIdx, *elimBuffer, *LNextSpaceInCol;
#  QDLDL_float *yVals;
#  QDLDL_float yVals_cidx;
#  QDLDL_bool  *yMarkers;
#  QDLDL_int   positiveValuesInD = 0;
#
#  //partition working memory into pieces
#  yMarkers        = bwork;
#  yIdx            = iwork;
#  elimBuffer      = iwork + n;
#  LNextSpaceInCol = iwork + n*2;
#  yVals           = fwork;
#
#
#  Lp[0] = 0; //first column starts at index zero
#
#  for(i = 0; i < n; i++){
#
#    //compute L column indices
#    Lp[i+1] = Lp[i] + Lnz[i];   //cumsum, total at the end
#
#    // set all Yidx to be 'unused' initially
#    //in each column of L, the next available space
#    //to start is just the first space in the column
#    yMarkers[i]  = QDLDL_UNUSED;
#    yVals[i]     = 0.0;
#    D[i]         = 0.0;
#    LNextSpaceInCol[i] = Lp[i];
#  }
#
#  // First element of the diagonal D.
#  D[0]     = Ax[0];
#  if(D[0] == 0.0){return -1;}
#  if(D[0]  > 0.0){positiveValuesInD++;}
#  Dinv[0] = 1/D[0];
#
#  //Start from 1 here. The upper LH corner is trivially 0
#  //in L b/c we are only computing the subdiagonal elements
#  for(k = 1; k < n; k++){
#
#    //NB : For each k, we compute a solution to
#    //y = L(0:(k-1),0:k-1))\b, where b is the kth
#    //column of A that sits above the diagonal.
#    //The solution y is then the kth row of L,
#    //with an implied '1' at the diagonal entry.
#
#    //number of nonzeros in this row of L
#    nnzY = 0;  //number of elements in this row
#
#    //This loop determines where nonzeros
#    //will go in the kth row of L, but doesn't
#    //compute the actual values
#    tmpIdx = Ap[k+1];
#
#    for(i = Ap[k]; i < tmpIdx; i++){
#
#      bidx = Ai[i];   // we are working on this element of b
#
#      //Initialize D[k] as the element of this column
#      //corresponding to the diagonal place.  Don't use
#      //this element as part of the elimination step
#      //that computes the k^th row of L
#      if(bidx == k){
#        D[k] = Ax[i];
#        continue;
#      }
#
#      yVals[bidx] = Ax[i];   // initialise y(bidx) = b(bidx)
#
#      // use the forward elimination tree to figure
#      // out which elements must be eliminated after
#      // this element of b
#      nextIdx = bidx;
#
#      if(yMarkers[nextIdx] == QDLDL_UNUSED){   //this y term not already visited
#
#        yMarkers[nextIdx] = QDLDL_USED;     //I touched this one
#        elimBuffer[0]     = nextIdx;  // It goes at the start of the current list
#        nnzE              = 1;         //length of unvisited elimination path from here
#
#        nextIdx = etree[bidx];
#
#        while(nextIdx != QDLDL_UNKNOWN && nextIdx < k){
#          if(yMarkers[nextIdx] == QDLDL_USED) break;
#
#          yMarkers[nextIdx] = QDLDL_USED;   //I touched this one
#          elimBuffer[nnzE] = nextIdx; //It goes in the current list
#          nnzE++;                     //the list is one longer than before
#          nextIdx = etree[nextIdx];   //one step further along tree
#
#        } //end while
#
#        // now I put the buffered elimination list into
#        // my current ordering in reverse order
#        while(nnzE){
#          yIdx[nnzY++] = elimBuffer[--nnzE];
#        } //end while
#      } //end if
#
#    } //end for i
#
#    //This for loop places nonzeros values in the k^th row
#    for(i = (nnzY-1); i >=0; i--){
#
#      //which column are we working on?
#      cidx = yIdx[i];
#
#      // loop along the elements in this
#      // column of L and subtract to solve to y
#      tmpIdx = LNextSpaceInCol[cidx];
#      yVals_cidx = yVals[cidx];
#      for(j = Lp[cidx]; j < tmpIdx; j++){
#        yVals[Li[j]] -= Lx[j]*yVals_cidx;
#      }
#
#      //Now I have the cidx^th element of y = L\b.
#      //so compute the corresponding element of
#      //this row of L and put it into the right place
#      Li[tmpIdx] = k;
#      Lx[tmpIdx] = yVals_cidx *Dinv[cidx];
#
#      //D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
#      D[k] -= yVals_cidx*Lx[tmpIdx];
#      LNextSpaceInCol[cidx]++;
#
#      //reset the yvalues and indices back to zero and QDLDL_UNUSED
#      //once I'm done with them
#      yVals[cidx]     = 0.0;
#      yMarkers[cidx]  = QDLDL_UNUSED;
#
#    } //end for i
#
#    //Maintain a count of the positive entries
#    //in D.  If we hit a zero, we can't factor
#    //this matrix, so abort
#    if(D[k] == 0.0){return -1;}
#    if(D[k]  > 0.0){positiveValuesInD++;}
#
#    //compute the inverse of the diagonal
#    Dinv[k]= 1/D[k];
#
#  } //end for k
#
#  return positiveValuesInD;
#
#}

def extend_upper_():
    pass

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', '-t', nargs='?',
        default='_HPC_3x3_H2',
        help='Testcase to take data from.',
    )
    #parser.add_argument('--level','-l', action='store_true', help='Compute level vector derived by level scheduling the L matrix.')
    parser.add_argument('--etree','-e', action='store_true', help='Compute elimintation tree of KKT matrix.')
    parser.add_argument('--plot','-p', action='store_true', help='Plot L matrix.')
    args = parser.parse_args()

    # Parse Workspace
    problem = args.test
    wpfile = f'build/{problem}/workspace_orig.c'
    if not os.path.exists(wpfile):
        raise ValueError(f'Workspace File {wpfile} does not exist.')
    wp = WorkspaceParser(wpfile)

    # extract data
    n = wp.work.n
    m = wp.work.m
    Pp = wp.work.Pp
    Pi = wp.work.Pi
    Px = wp.work.Px
    P = sp.csc_matrix((Px,Pi,Pp),shape=(n,n))
    Ap = wp.work.Ap
    Ai = wp.work.Ai
    Ax = wp.work.Ax
    A = sp.csc_matrix((Ax,Ai,Ap),shape=(m,n))

    # Ruiz equilibriation (currently just extract from code generation)
    Dsc = wp.work.Dscaling
    Esc = wp.work.Escaling
    csc = wp.workspace['c_scaling']

    # Regularization
    Pbar = csc*P.multiply(Dsc**2)
    #Pbar only stores the lower part of the P matrix
    print(f'Pbar: {repr(Pbar)}')
    Abar = A
    print(f'Abar: {repr(Abar)}')

    #KKT matrix creation
    sigma = wp.work.sigma
    #rho = wp.work.rho
    rho_inv_vec = np.diag(1/wp.work.rho_vec)
    K = sp.bmat([[Pbar+sigma*sp.eye(n), Abar.transpose()],[Abar,-rho_inv_vec]])
    wp.work.K = K
    print(f'K: {repr(K)}')

    # permutation

    # decomposition
    #breakpoint()
    #print_LDLT(K.toarray(),n+m)
    #breakpoint()
    #Ld,Dd,perm = ldl(K.toarray())
    #L = sp.csc_matrix(Ld)
    #D = sp.csc_matrix(Dd)
    #print(f'L from scipy LDL: {repr(K)}')

    #breakpoint()

    if args.etree:
        compute_etree(K) 
    if args.plot:
        plot_schedule(L,schedule,cuts)





