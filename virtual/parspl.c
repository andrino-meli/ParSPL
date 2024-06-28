#include "workspace.h"
#include "parspl.h"
#include "runtime.h"
#include "scheduled_data.h"

#ifdef __RT_SSSR_ENABLE
#define SSSR
#endif

double * const bp_avoid_wall_bound_err = bp_tmp;
double * const bp_tmp_g = &bp_avoid_wall_bound_err[-N_CCS*FIRST_BP];

void solve(int core_id){
    asm volatile("_solve: \n":::);
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
                __RT_SEPERATOR
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
    asm volatile("_permute%=: \n":::);
    if (core_id < N_CCS){
        __RT_SSSR_BLOCK_BEGIN
        uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
        uint16_t* perm_start = &Perm[start_perm[core_id]];
        double* bp_start = &bp[start_perm[core_id]];
        asm volatile(
            __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)

            __RT_SSSR_SCFGWI(%[b], 0,       __RT_SSSR_REG_IDX_BASE)
            __RT_SSSR_SCFGWI(%[perm], 0,     __RT_SSSR_REG_RPTR_INDIR)

            __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_WPTR_0)
            "frep.o     %[len], 1, 1, 0      \n"
            "fmv.d        ft1, ft0             \n"
            :: [len]"r"(len), [bp]"r"(bp_start), [b]"r"(&b), [perm]"r"(perm_start)
            : "memory", "t0"
        );
    __RT_SSSR_BLOCK_END
    }
}
#else
void permute(int core_id) {
    //__RT_SEPERATOR for clean measurement have it outside.
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
    asm volatile("_permuteT: \n":::);
    if (core_id < N_CCS){
        __RT_SSSR_BLOCK_BEGIN
        uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
        uint16_t* perm_start = &Perm[start_perm[core_id]];
        double* bp_start = &bp[start_perm[core_id]];
        asm volatile(
            __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
            __RT_SSSR_SCFGWI(%[x], 0,        __RT_SSSR_REG_IDX_BASE)
            __RT_SSSR_SCFGWI(%[perm], 0,     __RT_SSSR_REG_WPTR_INDIR)
            "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
            "mv      zero, a6                    \n" //_rt_fpu_fence_full();
            "csrr    zero,0x7c2                  \n" // __rt_barrier();
            __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_RPTR_0)
            "csrr zero, mcycle                   \n"
            "frep.o     %[len], 1, 1, 0      \n"
            "fmv.d        ft0, ft1             \n"
            :: [len]"r"(len), [bp]"r"(bp_start), [x]"r"(&x), [perm]"r"(perm_start)
            : "memory", "zero", "a6", "fa1"
        );
        __RT_SSSR_BLOCK_END
    } else {
        __RT_SEPERATOR
    }
}
#else
void permuteT(int core_id) {
    __RT_SEPERATOR
    if (core_id < N_CCS){
        for (int i = core_id; i < LINSYS_N; i += N_CCS) {
            x[Perm[i]] = bp[i];
        }
    }
}
#endif


#ifdef SSSR
void diag_inv_mult(int core_id) {
    asm volatile("_diag_inv_mult: \n":::);
    // multiply
    __RT_SSSR_BLOCK_BEGIN
    uint32_t len = len_perm[core_id]; //len is 0 for length of 1: so store -1 explicitly
    double* bp_start = &bp[start_perm[core_id]];
    double* dinv_start = &Dinv[start_perm[core_id]];
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)

        __RT_SSSR_SCFGWI(%[bp], 0,       __RT_SSSR_REG_WPTR_0)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        __RT_SSSR_SCFGWI(%[bp], 1,       __RT_SSSR_REG_RPTR_0)
        __RT_SSSR_SCFGWI(%[dinv], 2,     __RT_SSSR_REG_RPTR_0)
        "csrr zero, mcycle                   \n"
        "frep.o     %[len], 1, 1, 0      \n"
        "fmul.d  ft0, ft1, ft2             \n"
        //bp[i] *= Dinv[i];
        :: [len]"r"(len), [bp]"r"(bp_start), [dinv]"r"(dinv_start)
        : "memory", "zero", "a6", "fa1"
    );
    // clear out bp_tmp
    //__rt_fpu_fence_full(); //avoidable by streaming on ft1
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[stride_ex], 31,   __RT_SSSR_REG_STRIDE_0)
        __RT_SSSR_SCFGWI(%[tmp], 1,       __RT_SSSR_REG_WPTR_0)
        "frep.o     %[len], 1, 1, 0      \n"
        "fmv.d        ft1, %[zero]             \n"
        __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)
        :: [len]"r"(LAST_BP-FIRST_BP-1), [tmp]"r"(&bp_tmp[core_id]),
           [stride]"r"(8), [zero]"f"(0.0), [stride_ex]"r"(8*N_CCS)
        : "memory"
    );
    __RT_SSSR_BLOCK_END
}
#else
void diag_inv_mult(int core_id) {
    // multiply
    __RT_SEPERATOR
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
    asm volatile("_diaginv_lsolve: \n":::);
    // iterate through the rows
    //asm volatile(
    //    __RT_SSSR_SCFGWI(%[len2],  2,     __RT_SSSR_REG_BOUND_0)
    //    __RT_SSSR_SCFGWI(%[bp_cp], 2,     __RT_SSSR_REG_IDX_BASE)
    //    __RT_SSSR_SCFGWI(%[row],   2,     __RT_SSSR_REG_WPTR_INDIR)
    //    :: [len2]"r"(s->num_rows-1), [bp_cp]"r"(&bp_cp[s->rowa]), [row]"r"(s->assigned_rows)
    //    : "memory"
    //);
    __RT_SSSR_BLOCK_BEGIN
    __RT_SEPERATOR
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int row = s->assigned_rows[i];
        // dot product of row and bp
        double val;
        asm volatile(
            "fmv.d ft3, %[zero]             \n"
            "fmv.d ft4, %[zero]             \n"
            "fmv.d ft5, %[zero]             \n"
            "fmv.d ft6, %[zero]             \n"
            __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
            // TODO: revert access to avoid TCDM congestion
            __RT_SSSR_SCFGWI(%[bp] ,   0,     __RT_SSSR_REG_RPTR_0)
            __RT_SSSR_SCFGWI(%[mat],   1,     __RT_SSSR_REG_RPTR_0)
            "frep.o    %[len], 1, 3, 0b1001 \n"
            "fmadd.d    ft3, ft0, ft1, ft3  \n"
            "fadd.d     ft5, ft5, ft6       \n"
            "fadd.d     ft3, ft3, ft4       \n"
            "fadd.d     %[val], ft3, ft5    \n"
            : [val]"=f"(val)
            : [len]"r"(row), [bp]"r"(&bp[s->rowa]),
              [mat]"r"(&s->mat[row*s->n]), [zero]"f"(0.0)
            : "memory", "ft3","ft4","ft5","ft6"
        );
        bp_cp[s->rowa + row] = val;
    }
    // update bp from bp_cp
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
        __RT_SSSR_SCFGWI(%[row], 1,    __RT_SSSR_REG_RPTR_INDIR)
        "csrr zero, mcycle                   \n"
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
    __RT_SEPERATOR
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
    __RT_SEPERATOR
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
    asm volatile("_diaginv_ltsolve: \n":::);
    // The first row in FE and the last column in BS can be neglected
    // as multiplication with it is just the identity.
    // Therefor we only process the rows 1..n and columns 0..n-1
    // iterate through the rows
    asm volatile(
        __RT_SSSR_SCFGWI(%[stride_ex], 1,   __RT_SSSR_REG_STRIDE_0)
        :: [stride_ex]"r"(s->n*8) : "memory"
    );
    __RT_SSSR_BLOCK_BEGIN_NOSTART
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int col = s->n - s->assigned_rows[i] - 1; // row is saved in mat as col
                                        // -1 as we process 0..n-1
        double val;
        double const * const bparg = &bp[s->rowa + col];
        double const * const matarg = &s->mat[col*s->n + col];
        if ( i == 0 ){ // for i=0 also synchronize!
            asm volatile(
                // TODO: revert access to avoid TCDM congestion
                "fmv.d ft3, %[zero]             \n"
                "fmv.d ft4, %[zero]             \n"
                "fmv.d ft5, %[zero]             \n"
                "fmv.d ft6, %[zero]             \n"
                __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
                "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
                "mv      zero, a6                    \n" //_rt_fpu_fence_full();
                "csrr    zero,0x7c2                  \n" // __rt_barrier();
                __RT_SSSR_ENABLE
                __RT_SSSR_SCFGWI(%[bp] ,   0,     __RT_SSSR_REG_RPTR_0)
                __RT_SSSR_SCFGWI(%[mat],   1,     __RT_SSSR_REG_RPTR_0)
                "csrr zero, mcycle                   \n"
                "frep.o    %[len], 1, 3, 0b1001 \n"
                "fmadd.d    ft3, ft0, ft1, ft3  \n"
                // val += s->mat[row * s->n + col] * bp[s->rowa + row];
                "fadd.d     ft5, ft5, ft6       \n"
                "fadd.d     ft3, ft3, ft4       \n"
                "fadd.d     %[val], ft3, ft5    \n"
                : [val]"=f"(val)
                : [len]"r"(s->assigned_rows[i]), [bp]"r"(bparg),
                  [mat]"r"(matarg), [zero]"f"(0.0)
                : "memory", "ft3","ft4","ft5","ft6", "a6"
            );
            bp_cp[s->rowa + col] = val;
        } else {
            asm volatile(
                // TODO: revert access to avoid TCDM congestion
                "fmv.d ft3, %[zero]             \n"
                "fmv.d ft4, %[zero]             \n"
                "fmv.d ft5, %[zero]             \n"
                "fmv.d ft6, %[zero]             \n"
                __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
                __RT_SSSR_SCFGWI(%[bp] ,   0,     __RT_SSSR_REG_RPTR_0)
                __RT_SSSR_SCFGWI(%[mat],   1,     __RT_SSSR_REG_RPTR_0)
                "frep.o    %[len], 1, 3, 0b1001 \n"
                "fmadd.d    ft3, ft0, ft1, ft3  \n"
                // val += s->mat[row * s->n + col] * bp[s->rowa + row];
                "fadd.d     ft5, ft5, ft6       \n"
                "fadd.d     ft3, ft3, ft4       \n"
                "fadd.d     %[val], ft3, ft5    \n"
                : [val]"=f"(val)
                : [len]"r"(s->assigned_rows[i]), [bp]"r"(bparg),
                  [mat]"r"(matarg), [zero]"f"(0.0)
                : "memory", "ft3","ft4","ft5","ft6"
            );
            bp_cp[s->rowa + col] = val;
        }
    }
    asm volatile(
        __RT_SSSR_SCFGWI(%[stride], 1,   __RT_SSSR_REG_STRIDE_0)
        :: [stride]"r"(8) : "memory"
    );
    // update bp from bp_cp
    uint32_t len = s->num_rows-1;
    double* bp_start = &bp[s->rowa-1];
    double* bp_cp_start = &bp_cp[s->rowa-1];
    asm volatile(
        __RT_SSSR_SCFGWI(%[len], 31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp], 0,       __RT_SSSR_REG_IDX_BASE)
        __RT_SSSR_SCFGWI(%[bp_cp], 1,    __RT_SSSR_REG_IDX_BASE)

        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        __RT_SSSR_SCFGWI(%[row], 0,      __RT_SSSR_REG_WPTR_INDIR) //TODO move
        __RT_SSSR_SCFGWI(%[row], 1,      __RT_SSSR_REG_RPTR_INDIR)
        "csrr zero, mcycle                   \n"
        "frep.o     %[len], 1, 1, 0          \n"
        "fmv.d  ft0, ft1                     \n"
        //bp[i] *= Dinv[i];
        :: [len]"r"(len), [bp]"r"(bp_start), [bp_cp]"r"(bp_cp_start), [row]"r"(s->assigned_rows)
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
    __RT_SEPERATOR
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
    __RT_SEPERATOR
    // update bp from bp_cp
    for(unsigned int i = 0; i < s->num_rows; i++){
        unsigned int row = s->assigned_rows[i] - 1;
        bp[s->rowa + row] = bp_cp[s->rowa + row];
    }
}
#endif


#ifdef SSSR
void collist_lsolve(Collist const * s, int core_id) {
    asm volatile("_collist_lsolve: \n":::);
    // stream over bp[ri[..]] and rx[]
    if( s->num_data != 0) { // no empty stream
        __RT_SSSR_BLOCK_BEGIN
        asm volatile(
            __RT_SSSR_SCFGWI(%[len],      2,     __RT_SSSR_REG_BOUND_0)
            __RT_SSSR_SCFGWI(%[icfg2],     31,    __RT_SSSR_REG_IDX_CFG)
            __RT_SSSR_SCFGWI(%[bp_tmp_g], 31,    __RT_SSSR_REG_IDX_BASE)

            __RT_SSSR_SCFGWI(%[rx],  2,    __RT_SSSR_REG_RPTR_0)
            "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
            "mv      zero, a6                    \n" //_rt_fpu_fence_full();
            "csrr    zero,0x7c2                  \n" // __rt_barrier();
            "csrr zero, mcycle                   \n"
            __RT_SSSR_SCFGWI(%[icfg2],     31,    __RT_SSSR_REG_IDX_CFG)
            :: [len]"r"(s->num_data-1), [rx]"r"(s->rx),
               [icfg2]"r"(__RT_SSSR_IDX_CFG(__RT_SSSR_IDXSIZE_U16,LOG2_N_CCS,0)),
               [bp_tmp_g]"r"(&bp_tmp_g[core_id])
            : "memory", "zero", "a6", "fa1"
        );
        // work through columns
        // TODO: synchronize read access inbetween columns!!
        //       one might access same data
        unsigned int pos = 0;
        //for(unsigned int i = 0; i < s->num_cols; i++){
        //    unsigned int col = s->assigned_cols[i];
        //    double val = bp[col]; 
        //    for(unsigned int j = 0; j < s->len_cols[i]; j++){
        //        asm volatile(__RT_SSSR_ENABLE : "+f"(_rt_sssr_2), "+f"(_rt_sssr_1), "+f"(_rt_sssr_0)  :: "memory");
        //        bp_tmp_g[core_id + N_CCS * s->ri[pos]] -= val * _rt_sssr_2;
        //        pos++;
        //    }
        //}
        //for(unsigned int j = 0; j < s->len_cols[i]/2; j++){
        unsigned int i = 0;
        const uint16_t * const assigned_cols = s->assigned_cols;
        const uint16_t * const len_cols = s->len_cols;
        const uint16_t * const ri = s->ri;
        for( ;(i+1) < s->num_cols; ){
            unsigned int col = assigned_cols[i];
            double val = bp[col]; 
            asm volatile(
             __RT_SSSR_SCFGWI(%[ilen], 31,     __RT_SSSR_REG_BOUND_0)
             __RT_SSSR_SCFGWI(%[ri],    0,    __RT_SSSR_REG_WPTR_INDIR)
             __RT_SSSR_SCFGWI(%[ri],    1,    __RT_SSSR_REG_RPTR_INDIR)
            "frep.o    %[ilen], 1, 0, 0       \n"
            "fnmsub.d  ft0, %[val], ft2, ft1          \n"
            : [val]"+f"(val)
            : [ilen]"r"(len_cols[i]-1), [ri]"r"(&ri[pos])
            : "memory");
            pos += len_cols[i];
            i++;
            col = assigned_cols[i];
            val = bp[col]; 
            asm volatile(
             __RT_SSSR_SCFGWI(%[ilen], 31,     __RT_SSSR_REG_BOUND_0)
             __RT_SSSR_SCFGWI(%[ri],    1,    __RT_SSSR_REG_WPTR_INDIR)
             __RT_SSSR_SCFGWI(%[ri],    0,    __RT_SSSR_REG_RPTR_INDIR)
            "frep.o    %[ilen], 1, 0, 0       \n"
            "fnmsub.d  ft1, %[val], ft2, ft0          \n"
            : [val]"+f"(val)
            : [ilen]"r"(len_cols[i]-1), [ri]"r"(&ri[pos])
            : "memory");
            pos += len_cols[i];
            i++;
        }
        if(i < s->num_cols){
            unsigned int col = assigned_cols[i];
            double val = bp[col]; 
            asm volatile(
             __RT_SSSR_SCFGWI(%[ilen], 31,     __RT_SSSR_REG_BOUND_0)
             __RT_SSSR_SCFGWI(%[ri],    0,    __RT_SSSR_REG_WPTR_INDIR)
             __RT_SSSR_SCFGWI(%[ri],    1,    __RT_SSSR_REG_RPTR_INDIR)
            "frep.o    %[ilen], 1, 0, 0       \n"
            "fnmsub.d  ft0, %[val], ft2, ft1          \n"
            : [val]"+f"(val)
            : [ilen]"r"(len_cols[i]-1), [ri]"r"(&ri[pos])
            : "memory");
        }
        asm volatile(
            __RT_SSSR_SCFGWI(%[icfg],     31,    __RT_SSSR_REG_IDX_CFG)
            :: [icfg]"r"(__RT_SSSR_IDXSIZE_U16) : "memory"
        );
        __RT_SSSR_BLOCK_END
    } else {
        __RT_SEPERATOR 
    }
    // synchronize
    //__RT_SEPERATOR
    // reduce bp_tmp1 up to bp_tmp7 into bp
    //uint32_t threeUnlen = s->reductionlen / 3;
    asm volatile(
        //__RT_SSSR_SCFGWI(%[len],       2,    __RT_SSSR_REG_BOUND_0)
        //__RT_SSSR_SCFGWI(%[stride_ex], 31,   __RT_SSSR_REG_STRIDE_0)

        //__RT_SSSR_SCFGWI(%[bp_tmp_g0],  0     __RT_SSSR_REG_RPTR_0)
        //__RT_SSSR_SCFGWI(%[bp_tmp_g1],  1,    __RT_SSSR_REG_RPTR_0)
        //__RT_SSSR_SCFGWI(%[bp_tmp_g2],  2,    __RT_SSSR_REG_RPTR_0)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        "csrr    zero, mcycle                \n"
        __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)
        :: [len]"r"(N_CCS*s->reductionlen-1),
           [bp_tmp_g0]"r"(&bp_tmp_g[N_CCS*s->reductiona]),
           [bp_tmp_g1]"r"(&bp_tmp_g[N_CCS*s->reductiona+1]),
           [bp_tmp_g2]"r"(&bp_tmp_g[N_CCS*s->reductiona+2]),
           [stride]"r"(8),  [stride_ex]"r"(2*8)
        : "memory", "zero", "a6", "fa1"
    );
    unsigned int i = s-> reductiona;
    for(; i < s->reductiona + s->reductionlen; i++) {
        // adder tree
        // Assuming N_CCS = 8
        bp[i] += ((bp_tmp_g[N_CCS*i+0] + bp_tmp_g[N_CCS*i+1]) +
                  (bp_tmp_g[N_CCS*i+2] + bp_tmp_g[N_CCS*i+3])) +
                 ((bp_tmp_g[N_CCS*i+4] + bp_tmp_g[N_CCS*i+5]) +
                  (bp_tmp_g[N_CCS*i+6] + bp_tmp_g[N_CCS*i+7]));
    }
    //asm volatile(
    //    __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)
    //    :: [stride]"r"(8) : "memory"
    //);
}
#else
void collist_lsolve(Collist const * s, int core_id) {
    // pos array to index over ri, rx
    // TODO: when streaming add an if condition to circumvent an empty stream
    unsigned int pos = 0; 
    // work through columns
    __RT_SEPERATOR
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
    __RT_SEPERATOR
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
#endif

#ifdef SSSR
void collist_ltsolve(Collist const * s) {
    asm volatile("_collist_ltsolve: \n":::);
    __RT_SSSR_BLOCK_BEGIN
    // stream over bp[ri[..]] and rx[]
    asm volatile(
        __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp],  1,       __RT_SSSR_REG_IDX_BASE)

        __RT_SSSR_SCFGWI(%[rx],  2,    __RT_SSSR_REG_RPTR_0)
        "fmv.x.w a6, fa1                     \n" //_rt_fpu_fence_full();
        "mv      zero, a6                    \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2                  \n" // __rt_barrier();
        __RT_SSSR_SCFGWI(%[ri],  1,    __RT_SSSR_REG_RPTR_INDIR)
        "csrr zero, mcycle                   \n"
        :: [len]"r"(s->num_data-1), [bp]"r"(bp), [rx]"r"(s->rx), [ri]"r"(s->ri)
        : "memory", "zero", "a6", "fa1"
    );
    // work through rows
    for(unsigned int i = 0; i < s->num_cols; i++){
        unsigned int row = s->assigned_cols[i];
        // work through data in a row: read val
        double register val asm("ft3");
        double register val4 asm("ft4");
        double register val5 asm("ft5");
        double register val6 asm("ft6");
        val = bp[row]; //TODO: use single SSSR stream for this load
        val4 = 0.0;
        val5 = 0.0;
        val6 = 0.0;
        // update val
        asm volatile(
        //"fmv.d     ft4, %[zero]                \n;"
        //"fmv.d     ft5, %[zero]                \n;"
        "frep.o    %[ilen], 1, 3, 0b1001       \n"
        "fnmsub.d  ft3, ft1, ft2, ft3          \n"
        "fadd.d    ft3, ft3, ft4               \n"
        "fadd.d    ft5, ft5, ft6               \n"
        "fadd.d    ft3, ft3, ft5               \n"
        : [val]"+f"(val), [val4]"+f"(val4), [val5]"+f"(val5), [val6]"+f"(val6)
        : [ilen]"r"(s->len_cols[i]-1), [zero]"f"(0.0)
        : "memory");
        bp[row] = val;
    }
    __RT_SSSR_BLOCK_END
}
#else
void collist_ltsolve(Collist const * s) {
    // pos array to index over ri, rx
    int pos = 0;
    // work through rows
    __RT_SEPERATOR
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
    asm volatile("_mapping_lsolve: \n":::);
    uint32_t len = s->assigned_data -1;
    double* bp_cph = &bp_cp[start_perm[core_id]];
    __RT_SSSR_BLOCK_BEGIN
    asm volatile(
        __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp], 31,       __RT_SSSR_REG_IDX_BASE)
        // bp_cph[..] =  bp[col]*val
        "fmv.x.w a6, fa1                  \n" //_rt_fpu_fence_full();
        "mv      zero, a6                 \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2               \n" // __rt_barrier();
        __RT_SSSR_SCFGWI(%[col], 1,       __RT_SSSR_REG_RPTR_INDIR)
        __RT_SSSR_SCFGWI(%[bp_cp], 0,     __RT_SSSR_REG_WPTR_0)
        __RT_SSSR_SCFGWI(%[val], 2,       __RT_SSSR_REG_RPTR_0)
        "csrr zero, mcycle                   \n"
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
           [val]"r"(s->data), [row]"r"(s->ri)
        : "memory", "zero", "a6", "fa1"
    );
    __RT_SSSR_BLOCK_END
}
#else
void mapping_lsolve(Mapping const * s, int core_id) {
    __RT_SEPERATOR
    for(unsigned int i = 0; i < s->assigned_data; i++){
        uint16_t row = s->ri[i];
        uint16_t col = s->ci[i];
        double val = s->data[i];
        //bp[row] -= bp[col]*bp[col+1];
        bp[row] -= bp[col]*val;
    }
}
#endif


#ifdef SSSR
void mapping_ltsolve(Mapping const * s, int core_id) {
    asm volatile("_mapping_ltsolve: \n":::);
    uint32_t len = s->assigned_data -1;
    double* bp_cph = &bp_cp[start_perm[core_id]];
    __RT_SSSR_BLOCK_BEGIN
    asm volatile(
        __RT_SSSR_SCFGWI(%[len],  31,     __RT_SSSR_REG_BOUND_0)
        __RT_SSSR_SCFGWI(%[bp], 31,       __RT_SSSR_REG_IDX_BASE)
        // bp_cph[..] =  bp[col]*val
        "fmv.x.w a6, fa1                  \n" //_rt_fpu_fence_full();
        "mv      zero, a6                 \n" //_rt_fpu_fence_full();
        "csrr    zero,0x7c2               \n" // __rt_barrier();
        __RT_SSSR_SCFGWI(%[col], 1,       __RT_SSSR_REG_RPTR_INDIR)
        __RT_SSSR_SCFGWI(%[bp_cp], 0,     __RT_SSSR_REG_WPTR_0)
        __RT_SSSR_SCFGWI(%[val], 2,       __RT_SSSR_REG_RPTR_0)
        "csrr zero, mcycle                   \n"
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
        :: [len]"r"(len), [bp]"r"(bp), [bp_cp]"r"(bp_cph), [col]"r"(s->ri),
           [val]"r"(s->data), [row]"r"(s->ci)
        : "memory", "zero", "a6", "fa1"
    );
    __RT_SSSR_BLOCK_END
}
#else
void mapping_ltsolve(Mapping const * s, int core_id) {
    __RT_SEPERATOR
    for(unsigned int i = 0; i < s->assigned_data; i++){
        uint16_t col = s->ri[i];
        uint16_t row = s->ci[i];
        double val = s->data[i];
        bp[row] -= bp[col]*val;
    }
}
#endif
