#include "workspace.h"
#include "parspl.h"
#include "runtime.h"
#include "scheduled_data.h"

#ifdef __RT_SSSR_ENABLE
#define SSSR
#endif

double * const bp_tmp_g = &bp_tmp[-N_CCS*FIRST_BP];

void solve(int core_id){
    unsigned int argidx = argstruct_coreoffset[core_id];
    for(unsigned int enumidx = enum_coreoffset[core_id]; enumidx < enum_coreoffset[core_id+1]; enumidx++) {
        enum Kernel kern = enum_joined[enumidx];
        #ifdef VERBOSE
        //printf("H%d:\tkernel %d\t,argidx %d\n",core_id,kern,argidx);
        #endif
        switch (kern) {
            case COLLIST_LSOLVE:
                collist_lsolve((Collist *) argstruct_joined[argidx], core_id);
                argidx++;
                break;
            case COLLIST_LTSOLVE:
                collist_ltsolve((Collist *) argstruct_joined[argidx]);
                argidx++;
                break;
            case DIAGINV_LSOLVE:
                diaginv_lsolve((Diaginv *) argstruct_joined[argidx]);
                argidx++;
                break;
            case DIAGINV_LTSOLVE:
                diaginv_ltsolve((Diaginv *) argstruct_joined[argidx]);
                argidx++;
                break;
            case MAPPING_LSOLVE:
                mapping_lsolve((Mapping *) argstruct_joined[argidx], core_id);
                argidx++;
                break;
            case MAPPING_LTSOLVE:
                mapping_ltsolve((Mapping *) argstruct_joined[argidx], core_id);
                argidx++;
                break;
            case DIAG_INV_MULT:
                diag_inv_mult(core_id);
                break;
            case SYNCH:
                __rt_seperator();
                break;
            default:
                #ifdef VERBOSE
                printf("Error no case taken for kernel");
                #endif
                break;
        }
    }
}

// Permute b to bp (b permuted)
// TODO: make lin sys library: indirection copy
#ifdef SSSR
void permute(int core_id) {
    if (core_id < N_CCS){
        __RT_SSSR_BLOCK_BEGIN
        uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
        uint16_t* perm_start = &Perm[start_perm[core_id]];
        double* bp_start = &bp[start_perm[core_id]];
        asm volatile(
            __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)

            __RT_SSSR_SCFGWI(%[b], 0,       __RT_SSSR_REG_IDX_BASE)
            __RT_SSSR_SCFGWI(%[icfg], 0,     __RT_SSSR_REG_IDX_CFG)
            __RT_SSSR_SCFGWI(%[perm], 0,     __RT_SSSR_REG_RPTR_INDIR)

            __RT_SSSR_SCFGWI(%[stride], 1,   __RT_SSSR_REG_STRIDE_0)
            __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_WPTR_0)
            "frep.o     %[len], 1, 1, 0      \n"
            "fmv.d        ft1, ft0             \n"
            :: [len]"r"(len), [bp]"r"(bp_start), [b]"r"(&b), [perm]"r"(perm_start),
               [stride]"r"(8), [icfg]"r"(__RT_SSSR_IDXSIZE_U16)
            : "memory"
        );
    __RT_SSSR_BLOCK_END
    }
}
#else
void permute(int core_id) {
    //__rt_seperator(); for clean measurement have it outside.
    if (core_id < N_CCS){
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            bp[i] = b[Perm[i]];
        }
    }
}
#endif

// Permute back bp to x: x = P_ermute^T*bp
// TODO: make lin sys library
#ifdef SSSR
void permuteT(int core_id) {
    if (core_id < N_CCS){
        __RT_SSSR_BLOCK_BEGIN
        uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
        uint16_t* perm_start = &Perm[start_perm[core_id]];
        double* bp_start = &bp[start_perm[core_id]];
        asm volatile(
            __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)

            __RT_SSSR_SCFGWI(%[x], 0,       __RT_SSSR_REG_IDX_BASE)
            __RT_SSSR_SCFGWI(%[icfg], 0,     __RT_SSSR_REG_IDX_CFG)
            __RT_SSSR_SCFGWI(%[perm], 0,     __RT_SSSR_REG_WPTR_INDIR)

            __RT_SSSR_SCFGWI(%[stride], 1,   __RT_SSSR_REG_STRIDE_0)
            __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_RPTR_0)
            "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
            "mv      zero, a6                    \n" //_rt_fpu_fence_full();
            "csrr    zero,0x7c2                  \n" // __rt_barrier();
            "csrr zero, mcycle                   \n"
            "frep.o     %[len], 1, 1, 0      \n"
            "fmv.d        ft0, ft1             \n"
            :: [len]"r"(len), [bp]"r"(bp_start), [x]"r"(&x), [perm]"r"(perm_start),
               [stride]"r"(8), [icfg]"r"(__RT_SSSR_IDXSIZE_U16)
            : "memory", "zero", "a6", "fa1"
        );
        __RT_SSSR_BLOCK_END
    } else {
        __rt_seperator();
    }
}
#else
void permuteT(int core_id) {
    __rt_seperator();
    if (core_id < N_CCS){
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            x[Perm[i]] = bp[i];
        }
    }
}
#endif


#ifdef SSSR
void diag_inv_mult(int core_id) {
    // multiply
    __RT_SSSR_BLOCK_BEGIN
    uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
    double* bp_start = &bp[start_perm[core_id]];
    double* dinv_start = &Dinv[start_perm[core_id]];
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)

        __RT_SSSR_SCFGWI(%[bp], 0,       __RT_SSSR_REG_WPTR_0)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        "csrr zero, mcycle                   \n"
        __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_RPTR_0)
        __RT_SSSR_SCFGWI(%[dinv], 2,     __RT_SSSR_REG_RPTR_0)
        "frep.o     %[len], 1, 1, 0      \n"
        "fmul.d  ft0, ft1, ft2             \n"
        //bp[i] *= Dinv[i];
        :: [len]"r"(len), [bp]"r"(bp_start), [dinv]"r"(dinv_start), [stride]"r"(8)
        : "memory", "zero", "a6", "fa1"
    );
    // clear out bp_tmp
    //__rt_fpu_fence_full(); //avoidable by streaming on ft1
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)
        __RT_SSSR_SCFGWI(%[tmp], 1,       __RT_SSSR_REG_WPTR_0)
        "frep.o     %[len], 1, 1, 0      \n"
        "fmv.d        ft1, %[zero]             \n"
        // bp_tmp[i] = 0;
        :: [len]"r"(LAST_BP-FIRST_BP-1), [tmp]"r"(&bp_tmp[core_id]),
           [stride]"r"(8*N_CCS), [zero]"f"(0.0)
        : "memory"
    );
    __RT_SSSR_BLOCK_END
}
#else
void diag_inv_mult(int core_id) {
    // multiply
    __rt_seperator();
    for (int i = core_id; i < LINSYS_N; i += N_CCS) {
        bp[i] *= Dinv[i];
    }
    // clear out bp_tmp
    // Below for loop also works but potentially produces bank conflicts
    //for(int i = 0; i < LAST_BP-FIRST_BP; i++) {
    //    bp_tmp[(LAST_BP-FIRST_BP)*core_id + i] = 0;
    //}
    // interleaved acces version is presumably better against bank conflicts
    for(int i = core_id; i < N_CCS*(LAST_BP-FIRST_BP); i+=N_CCS) {
        bp_tmp[i] = 0;
    }
}
#endif


#ifdef SSSR
void diaginv_lsolve(Diaginv const * s){
    // iterate through the rows
    __rt_seperator();
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
    // update bp from bp_cp
    __RT_SSSR_BLOCK_BEGIN
    uint32_t len = s->num_rows-1;
    double* bp_start = &bp[s->rowa];
    double* bp_cp_start = &bp_cp[s->rowa];
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp], 0,       __RT_SSSR_REG_IDX_BASE)
        __RT_SSSR_SCFGWI(%[bp_cp], 1,       __RT_SSSR_REG_IDX_BASE)

        __RT_SSSR_SCFGWI(%[row], 0,     __RT_SSSR_REG_WPTR_INDIR)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        "csrr zero, mcycle                   \n"
        __RT_SSSR_SCFGWI(%[row], 1,    __RT_SSSR_REG_RPTR_INDIR)
        "frep.o     %[len], 1, 1, 0      \n"
        "fmv.d  ft0, ft1                 \n"
        //bp[i] *= Dinv[i];
        :: [len]"r"(len), [bp]"r"(bp_start), [bp_cp]"r"(bp_cp_start),
           [row]"r"(s->assigned_rows)
        : "memory", "zero", "a6", "fa1"
    );
    __RT_SSSR_BLOCK_END
}
#else
void diaginv_lsolve(Diaginv const * s){
    // iterate through the rows
    __rt_seperator();
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
#endif


#ifdef SSSR
void diaginv_ltsolve(Diaginv const * s){
    // The first row in FE and the last column in BS can be neglected
    // as multiplication with it is just the identity.
    // Therefor we only process the rows 1..n and columns 0..n-1
    // iterate through the rows
    __rt_seperator();
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
    // update bp from bp_cp
    __RT_SSSR_BLOCK_BEGIN
    uint32_t len = s->num_rows-1;
    double* bp_start = &bp[s->rowa-1];
    double* bp_cp_start = &bp_cp[s->rowa-1];
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp], 0,       __RT_SSSR_REG_IDX_BASE)
        __RT_SSSR_SCFGWI(%[bp_cp], 1,       __RT_SSSR_REG_IDX_BASE)

        __RT_SSSR_SCFGWI(%[row], 0,     __RT_SSSR_REG_WPTR_INDIR)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        "csrr zero, mcycle                   \n"
        __RT_SSSR_SCFGWI(%[row], 1,    __RT_SSSR_REG_RPTR_INDIR)
        "frep.o     %[len], 1, 1, 0      \n"
        "fmv.d  ft0, ft1                 \n"
        //bp[i] *= Dinv[i];
        :: [len]"r"(len), [bp]"r"(bp_start), [bp_cp]"r"(bp_cp_start),
           [row]"r"(s->assigned_rows)
        : "memory", "zero", "a6", "fa1"
    );
    __RT_SSSR_BLOCK_END
}
#else
void diaginv_ltsolve(Diaginv const * s){
    // The first row in FE and the last column in BS can be neglected
    // as multiplication with it is just the identity.
    // Therefor we only process the rows 1..n and columns 0..n-1
    // iterate through the rows
    __rt_seperator();
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
#endif


void collist_lsolve(Collist const * s, int core_id) {
    // pos array to index over ri, rx
    // TODO: when streaming add an if condition to circumvent an empty stream
    unsigned int pos = 0; 
    // work through columns
    __rt_seperator();
    for(unsigned int i = 0; i < s->num_cols; i++){
        unsigned int col = s->assigned_cols[i];
        // access val to muliply column with
        double val = bp[col];
        // work through data in a column
        for(unsigned int j = 0; j < s->len_cols[i]; j++){
            // TODO: N_CCS interleaved data access is expensive: consider placing
            //  bp_tmp differently in memory to make access continuous
            bp_tmp_g[core_id + N_CCS * s->ri[pos]] -= val * s->rx[pos];
            pos++;
        }
    }
    // synchronize
    __rt_seperator();
    // reduce bp_tmp1 up to bp_tmp7 into bp
    for(int i = s->reductiona; i < s->reductiona + s->reductionlen; i++) {
        // adder tree
        // Assuming N_CCS = 8
        bp[i] += ((bp_tmp_g[N_CCS*i+0] + bp_tmp_g[N_CCS*i+1]) +
                  (bp_tmp_g[N_CCS*i+2] + bp_tmp_g[N_CCS*i+3])) +
                 ((bp_tmp_g[N_CCS*i+4] + bp_tmp_g[N_CCS*i+5]) +
                  (bp_tmp_g[N_CCS*i+6] + bp_tmp_g[N_CCS*i+7]));
    }
}


#ifdef SSSR
void collist_ltsolve(Collist const * s) {
    // pos array to index over ri, rx
    int pos = 0;
    // work through rows
    __rt_seperator();
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
#else
void collist_ltsolve(Collist const * s) {
    // pos array to index over ri, rx
    int pos = 0;
    // work through rows
    __rt_seperator();
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
#endif

#ifdef SSSR
// sadly we would need 4 SSSR streams to make this work fully!
// instead we stream twice and store intermed. in bp_cph
void mapping_lsolve(Mapping const * s, int core_id) {
    uint32_t len = s->assigned_data -1;
    double* bp_cph = &bp_cp[start_perm[core_id]];
    __RT_SSSR_BLOCK_BEGIN
    asm volatile(
        __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[icfg], 31,     __RT_SSSR_REG_IDX_CFG)
        __RT_SSSR_SCFGWI(%[bp], 31,       __RT_SSSR_REG_IDX_BASE)
        __RT_SSSR_SCFGWI(%[stride], 31,    __RT_SSSR_REG_STRIDE_0)
        // bp_cph[..] =  bp[col]*val
        "fmv.x.w a6, fa1                  \n" //_rt_fpu_fence_full();
        "mv      zero, a6                 \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2               \n" // __rt_barrier();
        "csrr zero, mcycle                   \n"
        __RT_SSSR_SCFGWI(%[col], 1,       __RT_SSSR_REG_RPTR_INDIR)
        __RT_SSSR_SCFGWI(%[bp_cp], 0,     __RT_SSSR_REG_WPTR_0)
        __RT_SSSR_SCFGWI(%[val], 2,       __RT_SSSR_REG_RPTR_0)
        "frep.o     %[len], 1, 1, 0       \n"
        "fmul.d ft0,ft1,ft2               \n"
        // bp[row] = bp[row] - bp_cph[..]
        // avoid fpu barrier by changing ft0 write stream to read stream
        __RT_SSSR_SCFGWI(%[row], 0,     __RT_SSSR_REG_RPTR_INDIR)
        __RT_SSSR_SCFGWI(%[row], 1,     __RT_SSSR_REG_WPTR_INDIR)
        "fmv.x.w a6, fa1                  \n" //_rt_fpu_fence_full();
        "mv      zero, a6                 \n" //_rt_fpu_fence_full();
        __RT_SSSR_SCFGWI(%[bp_cp], 2,   __RT_SSSR_REG_RPTR_0)
        "frep.o     %[len], 1, 1, 0     \n"
        "fsub.d ft1,ft0,ft2             \n"
        :: [len]"r"(len), [bp]"r"(bp), [bp_cp]"r"(bp_cph), [col]"r"(s->ci),
           [val]"r"(s->data), [row]"r"(s->ri),
           [stride]"r"(8), [icfg]"r"(__RT_SSSR_IDXSIZE_U16)
        : "memory", "zero", "a6", "fa1"
    );
    __RT_SSSR_BLOCK_END
    //for(int i = 0; i < s->assigned_data; i++){
    //    uint16_t row = s->ri[i];
    //    bp[row] -= bp_cph[i];
    //}
}
#else
void mapping_lsolve(Mapping const * s, int core_id) {
    __rt_seperator();
    for(unsigned int i = 0; i < s->assigned_data; i++){
        uint16_t row = s->ri[i];
        uint16_t col = s->ci[i];
        double val = s->data[i];
        bp[row] -= bp[col]*val;
    }
}
#endif

void mapping_ltsolve(Mapping const * s, int core_id) {
    __rt_seperator();
    for(unsigned int i = 0; i < s->assigned_data; i++){
        uint16_t col = s->ri[i];
        uint16_t row = s->ci[i];
        double val = s->data[i];
        bp[row] -= bp[col]*val;
    }
}
