#include "workspace.h"
#include "kernel.h"
#include "runtime.h"

void diaginv_lsolve(
/* 
 * Solves: x = mat*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */
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
