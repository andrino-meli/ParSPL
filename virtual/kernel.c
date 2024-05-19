#include "workspace.h"
#include "kernel.h"
#include "runtime.h"


#ifdef VERBOSE
void print_bp(int core_id){
    if(core_id == 0){
        printf("bp = [");
        for(int i = 0; i < LINSYS_N; i++){
            printf("r%d\t%f\n",i,bp[i]);
        }
        printf("]\n");
    }
}
#endif

// Permute b to bp (b permuted)
// TODO: make lin sys library: indirection copy
void permute(int core_id) {
    for (int i = core_id; i < LINSYS_N; i += N_CCS) {
        bp[i] = b[Perm[i]];
    }
    //__rt_fpu_fence_full();
    __rt_barrier();
}

// Permute back bp to x: x = P_ermute^T*bp
// TODO: make lin sys library
void permuteT(int core_id) {
    for (int i = core_id; i < LINSYS_N; i += N_CCS) {
        x[Perm[i]] = bp[i];
    }
    __rt_barrier();
}


// TODO: make lin sys library
static inline void empty_out_bp_tmp(double* bp_tmp) {
    for(int i = 0; i < LINSYS_N; i++) {
        bp_tmp[i] = 0;
    }
}

void diag_inv_mult(int core_id) {
    // multiply
    for (int i = core_id; i < LINSYS_N; i += N_CCS) {
        bp[i] *= Dinv[i];
    }
    // clear out bp_tmp
    switch (core_id){
        case 0:
            empty_out_bp_tmp(bp_tmp0);
            break;
        case 1:
            empty_out_bp_tmp(bp_tmp1);
            break;
        case 2:
            empty_out_bp_tmp(bp_tmp2);
            break;
        case 3:
            empty_out_bp_tmp(bp_tmp3);
            break;
        case 4:
            empty_out_bp_tmp(bp_tmp4);
            break;
        case 5:
            empty_out_bp_tmp(bp_tmp5);
            break;
        case 6:
            empty_out_bp_tmp(bp_tmp6);
            break;
        case 7:
            empty_out_bp_tmp(bp_tmp7);
            break;
        default:
            printf("Error: wrong core count configuration in code generation.");
            break;
    }
}

void diaginv_lsolve(
    unsigned int n,
    unsigned int rowa, // row/column offset
    double mat[], // invrse matrix
    uint16_t assigned_rows[],
    unsigned int num_rows
){
    // iterate through the rows
    for(unsigned int i = 0; i < num_rows; i++){
        unsigned int row = assigned_rows[i];
        // dot product of row and bp
        double val = 0;
        for(unsigned col = 0; col <= row; col++){
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
    for(unsigned int i = 0; i < num_rows; i++){
        unsigned int row = assigned_rows[i];
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
    for(unsigned int i = 0; i < num_rows; i++){
        unsigned int col = assigned_rows[i] - 1; // row is saved in mat as col
                                        // -1 as we process 0..n-1
        // dot product of col and bp
        double val = 0;
        for(unsigned int row = col; row < n; row++){
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
    for(unsigned int i = 0; i < num_rows; i++){
        unsigned int row = assigned_rows[i] - 1;
        bp[rowa + row] = bp_cp[rowa + row];
    }
}


void collist_lsolve(
    uint16_t num_cols, // number of columns
    uint16_t assigned_cols[], // list of assigned columns
    uint16_t len_cols[], // length of each column
    uint16_t num_data, // length of ri,rx
    uint16_t ri[], // row index
    double rx[], // row data
                 //
    double bp_tmp[],        // temporary bp vector
    uint16_t reductiona,    // reduction offset in bp_tmpH
    uint16_t reductionlen   // reduction length off bp_tmpH
){
    // pos array to index over ri, rx
    // TODO: when streaming add an if condition to circumvent an empty stream
    unsigned int pos = 0; 
    // work through columns
    for(unsigned int i = 0; i < num_cols; i++){
        unsigned int col = assigned_cols[i];
        // access val to muliply column with
        double val = bp[col];
        // work through data in a column
        for(unsigned int j = 0; j < len_cols[i]; j++){
            bp_tmp[ri[pos]] -= val*rx[pos];
            pos++;
        }
    }
    // synchronize
    __rt_barrier();
    // reduce bp_tmp1 up to bp_tmp7 into bp
    for(int i = reductiona; i < reductiona + reductionlen; i++) {
        // adder tree
        bp[i] += (bp_tmp0[i] + bp_tmp1[i]) + (bp_tmp2[i]+ bp_tmp3[i]) +
                (bp_tmp4[i] + bp_tmp5[i]) + (bp_tmp6[i]+ bp_tmp7[i]);
    }
}


void collist_ltsolve(
    uint16_t num_cols, // number of columns
    uint16_t assigned_cols[], // list of assigned columns
    uint16_t len_cols[], // length of each column
    uint16_t num_data, // length of ri,rx
    uint16_t ri[], // row index
    double rx[] // row data
) {
    // pos array to index over ri, rx
    int pos = 0;
    // work through rows
    for(unsigned int i = 0; i < num_cols; i++){
        unsigned int row = assigned_cols[i];
        // work through data in a row: read val
        double val = bp[row];
        // update val
        for(unsigned int j = 0; j < len_cols[i]; j++){
            val -= bp[ri[pos]]*rx[pos];
            pos++;
        }
        // write back
        bp[row] = val;
    }
}
