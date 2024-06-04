#include "workspace.h"
#include "parspl.h"
#include "runtime.h"
#include "scheduled_data.h"

void solve(int core_id){
    unsigned int argidx = argstruct_coreoffset[core_id];
    for(unsigned int enumidx = enum_coreoffset[core_id]; enumidx < enum_coreoffset[core_id+1]; enumidx++) {
        enum Kernel kern = enum_joined[enumidx];
        #ifdef VERBOSE
        printf("H%d:\tkernel %d\t,argidx %d\n",core_id,kern,argidx);
        #endif
        switch (kern) {
            case COLLIST_LSOLVE:
                collist_lsolve((Collist *) argstruct_joined[argidx]);
                argidx++;
                __rt_seperator();
                break;
            case COLLIST_LTSOLVE:
                collist_ltsolve((Collist *) argstruct_joined[argidx]);
                argidx++;
                __rt_seperator();
                break;
            case DIAGINV_LSOLVE:
                diaginv_lsolve((Diaginv *) argstruct_joined[argidx]);
                argidx++;
                __rt_seperator();
                break;
            case DIAGINV_LTSOLVE:
                diaginv_ltsolve((Diaginv *) argstruct_joined[argidx]);
                argidx++;
                __rt_seperator();
                break;
            case DIAG_INV_MULT:
                diag_inv_mult(core_id);
                break;
            case SYNCH:
                __rt_seperator();
                break;
            default:
                printf("Error no case taken for kernel");
        }
    }
}

// Permute b to bp (b permuted)
// TODO: make lin sys library: indirection copy
void permute(int core_id) {
    if (core_id < N_CCS){
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            bp[i] = b[Perm[i]];
        }
    }
    __rt_seperator();
}

// Permute back bp to x: x = P_ermute^T*bp
// TODO: make lin sys library
void permuteT(int core_id) {
    if (core_id < N_CCS){
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            x[Perm[i]] = bp[i];
        }
    }
    __rt_seperator();
}


// TODO: make lin sys library
static inline void empty_out_bp_tmp(double* bp_tmp) {
    for(int i = 0; i < LINSYS_N; i++) {
        bp_tmp[i] = 0;
    }
}

void diag_inv_mult(int core_id) {
    // multiply
    if(core_id < N_CCS) {
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            bp[i] *= Dinv[i];
        }
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
            #ifdef PRINTF
            printf("Error: wrong core count configuration in code generation.");
            #endif
            break;
    }
    __rt_seperator();
}

void diaginv_lsolve(Diaginv* s){
    // iterate through the rows
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int row = s->assigned_rows[i];
        //printf("diaginv_lsolve: processing row %d, row+rowa %d\n",row,row+s->rowa);
        // dot product of row and bp
        double val = 0;
        for(unsigned col = 0; col <= row; col++){
            // TODO: currently all harts have the same access pattern to bp
            //       have to reverse to avoid tcdm access congestion
            val += s->mat[row * s->n + col] * bp[s->rowa + col];
        }
        // update bp_cp[row]
        bp_cp[s->rowa + row] = val;
    }
    // synchronize
    __rt_seperator();
    // update bp from bp_cp
    // TODO: indirection copy
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int row = s->assigned_rows[i];
        bp[s->rowa + row] = bp_cp[s->rowa + row];
    }
}


void diaginv_ltsolve(Diaginv* s){
    // The first row in FE and the last column in BS can be neglected
    // as multiplication with it is just the identity.
    // Therefor we only process the rows 1..n and columns 0..n-1
    // iterate through the rows
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int col = s->n - s->assigned_rows[i] - 1; // row is saved in mat as col
                                        // -1 as we process 0..n-1
        //printf("diaginv_ltsolve: processing col %d, col+rowa %d\n",col,col+rowa);
        // dot product of col and bp
        double val = 0;
        for(unsigned int row = col; row < s->n; row++){
            val += s->mat[row * s->n + col] * bp[s->rowa + row];
            //printf("col %d row %d   s->mat[%d] bp[%d]\t",col, row, row*s->n+col,s->rowa + row);
            //printf("mat=%f\tbp=%f\tval=%f\n", s->mat[row * s->n + col], bp[s->rowa + row], val);
        }
        // update bp_cp[col]
        bp_cp[s->rowa + col] = val;
    }
    // synchronize
    __rt_seperator();
    // update bp from bp_cp
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int row = s->assigned_rows[i] - 1;
        bp[s->rowa + row] = bp_cp[s->rowa + row];
    }
}


void collist_lsolve(Collist* s){
    // pos array to index over ri, rx
    // TODO: when streaming add an if condition to circumvent an empty stream
    unsigned int pos = 0; 
    // work through columns
    for(unsigned int i = 0; i < s->num_cols; i++){
        unsigned int col = s->assigned_cols[i];
        // access val to muliply column with
        double val = bp[col];
        // work through data in a column
        for(unsigned int j = 0; j < s->len_cols[i]; j++){
            s->bp_tmp[s->ri[pos]] -= val*s->rx[pos];
            pos++;
        }
    }
    // synchronize
    __rt_seperator();
    // reduce bp_tmp1 up to bp_tmp7 into bp
    for(int i = s->reductiona; i < s->reductiona + s->reductionlen; i++) {
        // adder tree
        bp[i] += (bp_tmp0[i] + bp_tmp1[i]) + (bp_tmp2[i]+ bp_tmp3[i]) +
                (bp_tmp4[i] + bp_tmp5[i]) + (bp_tmp6[i]+ bp_tmp7[i]);
    }
}


void collist_ltsolve(Collist* s) {
    // pos array to index over ri, rx
    int pos = 0;
    // work through rows
    for(unsigned int i = 0; i < s->num_cols; i++){
        unsigned int row = s->assigned_cols[i];
        // work through data in a row: read val
        double val = bp[row];
        // update val
        for(unsigned int j = 0; j < s->len_cols[i]; j++){
            val -= bp[s->ri[pos]]*s->rx[pos];
            pos++;
        }
        // write back
        bp[row] = val;
    }
}
