#ifndef  KERNEL_H
#define KERNEL_H
#include <stdint.h>
#include "types.h"

// parspl linear system solution
void solve(int core_id);

// single core linear system solution using CSC matrix format
void solve_csc();

// parallel linear system solution using CSC matrix format
void psolve_csc(int core_id);

// parallel linear system solution using CSC matrix format employing level scheduling
//void perm_psolve_csc(int core_id);

// SSSR acceleration of parallel linear system solution using CSC matrix format
// (basically the state of my master thesis)
void sssr_psolve_csc(int core_id);
void SSSR_PQDLDL_Lsolve(uint32_t core_id);
void SSSR_PQDLDL_Ltsolve(uint32_t core_id);

typedef struct {
    const unsigned int n;         // shape(mat) = (n,n)
    const unsigned int rowa;      // row/column offset
    const FLOAT * const mat;            // inverse matrix
    const uint16_t * const assigned_rows;
    const unsigned int num_rows;  // len(assigned_rows)
} Diaginv;

typedef struct {
    const unsigned int n;         // shape(mat) = (n,n)
    const unsigned int rowa;      // row/column offset
    const FLOAT * const fold;            // inverse matrix, folded
    const uint16_t * const assigned_rows;
    const unsigned int num_rows;  // len(assigned_rows)
} Fold;

typedef struct {
    const uint16_t * const assigned_cols;
    const uint16_t * const len_cols;   // length of each column
    const uint16_t * const ri;           // row index
    const FLOAT * const rx;             // row data
    const uint16_t num_cols;      // number of columns
    const uint16_t num_data;      // length of ri,rx
    const uint16_t reductiona;    // reduction offset in bp_tmpH
    const uint16_t reductionlen;   // reduction length off bp_tmpH
} Collist;

typedef struct {
    const uint16_t * const ri;
    const uint16_t * const ci;
    const FLOAT * const data;
    const uint16_t assigned_data;
} Mapping;


void permute(int core_id);


void permuteT(int core_id);


void diaginv_lsolve(Diaginv const * s);
/* 
 * Solves: x = mat*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void diaginv_ltsolve(Diaginv const * s);
/* 
 * Solves: x = mat^T*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void fold_lsolve(Fold const * s);
/* 
 * Solves: x = fold*bp
 *         and stores x in bp_cp
 *         fold however is stored folded:
 *         so a more complex data access is required
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void fold_ltsolve(Fold const * s);
/* 
 * Solves: x = fold^T*bp
 *         and stores x in bp_cp
 *         fold however is stored folded:
 *         so a more complex data access is required
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void diag_inv_mult(int core_id);
/* vector vector multiplication
 *   bp[] = bp[]*Dinv[]
 *   clear out data
 */


void collist_lsolve(Collist const * s, int core_id);
/* 
 * Solves: Cx = bp
 *         and stores x in bp_tmp_h
 * Synchr: synchronizes cores
 * Copies: reduces bp_tmp_h0 up to bp_tmp_h7
 */

void collist_ltsolve(Collist const * s);
/* 
 * Solves: C^Tx = bp
 */

void mapping_lsolve(Mapping const * s, int core_id);
/* 
 * Solves: Mx = bp inplace
 */

void mapping_ltsolve(Mapping const * s, int core_id);
/* 
 * Solves: M^Tx = bp inplace
 */

#endif
