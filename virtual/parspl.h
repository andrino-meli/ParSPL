#ifndef  KERNEL_H
#define KERNEL_H
#include <stdint.h>

void solve();

typedef struct {
    const unsigned int n;         // shape(mat) = (n,n)
    const unsigned int rowa;      // row/column offset
    const double * const mat;            // inverse matrix
    const uint16_t * const assigned_rows;
    const unsigned int num_rows;  // len(assigned_rows)
} Diaginv;

typedef struct {
    const uint16_t * const assigned_cols;
    const uint16_t * const len_cols;   // length of each column
    const uint16_t * const ri;           // row index
    const double * const rx;             // row data
    const uint16_t num_cols;      // number of columns
    const uint16_t num_data;      // length of ri,rx
    const uint16_t reductiona;    // reduction offset in bp_tmpH
    const uint16_t reductionlen;   // reduction length off bp_tmpH
} Collist;

typedef struct {
    const uint16_t * const ri;
    const uint16_t * const ci;
    const double * const data;
    const uint16_t assigned_data;
} Mapping;


void permute();


void permuteT();


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

void diag_inv_mult();
/* vector vector multiplication
 *   bp[] = bp[]*Dinv[]
 *   clear out data
 */


void collist_lsolve(Collist const * s);
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

void mapping_lsolve(Mapping const * s);
/* 
 * Solves: Mx = bp inplace
 */

void mapping_ltsolve(Mapping const * s);
/* 
 * Solves: M^Tx = bp inplace
 */

#endif
