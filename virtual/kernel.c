#include "workspace.h"
#include "kernel.h"
#include "runtime.h"

void diaginv_lsolve(
    unsigned int n,
    unsigned int rowa, // row/column offset
    double mat[], // invrse matrix
    uint16_t assigned_rows[],
    unsigned int num_rows
){
    // iterate through the rows
    for(int i = 0; i < num_rows; i++){
        int row = assigned_rows[i];
        // dot product of row and bp
        double val = 0;
        for(int col = 0; col <= row; col++){
            // TODO: currently all harts have the same access pattern to bp
            //       have to reverse to avoid tcdm access congestion
            val += mat[row * n + col] * bp[rowa + col];
        }
        // update bp_cp[row]
        bp_cp[rowa + row] = val;
    }
    // synchronize
    __rt_barrier();
    // update bp from bp_cp
    // TODO: indirection copy
    for(int i = 0; i < num_rows; i++){
        int row = assigned_rows[i];
        bp[rowa + row] = bp_cp[rowa + row];
    }
}


void diaginv_ltsolve(
    unsigned int n, // shape(mat) = (n,n)
    unsigned int rowa, // row/column offset
    double mat[], // invrse matrix
    uint16_t assigned_rows[],
    unsigned int num_rows // len(assigned_rows)
    // The first row in FE and the last column in BS can be neglected
    // as multiplication with it is just the identity.
    // Therefor we only process the rows 1..n and columns 0..n-1
){
    // iterate through the rows
    for(int i = 0; i < num_rows; i++){
        int col = assigned_rows[i] - 1; // row is saved in mat as col
                                        // -1 as we process 0..n-1
        // dot product of col and bp
        double val = 0;
        for(int row = col; row < n; row++){
            val += mat[row * n + col] * bp[rowa + row];
            //printf("col %d row %d   mat[%d] bp[%d]\t",col, row, row*n+col,rowa + row);
            //printf("mat=%f\tbp=%f\tval=%f\n", mat[row * n + col], bp[rowa + row], val);
        }
        // update bp_cp[col]
        bp_cp[rowa + col] = val;
    }
    // synchronize
    __rt_barrier();
    // update bp from bp_cp
    for(int i = 0; i < num_rows; i++){
        int row = assigned_rows[i] - 1;
        bp[rowa + row] = bp_cp[rowa + row];
    }
}
