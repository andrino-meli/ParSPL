#ifndef  KERNEL_H
#define KERNEL_H
#include <stdint.h>

void solve(int core_id);

typedef struct {
    unsigned int n;         // shape(mat) = (n,n)
    unsigned int rowa;      // row/column offset
    double* mat;            // inverse matrix
    uint16_t* assigned_rows;
    unsigned int num_rows;  // len(assigned_rows)
} Diaginv;

typedef struct {
    uint16_t* assigned_cols;
    uint16_t* len_cols;   // length of each column
    uint16_t* ri;           // row index
    double* rx;             // row data
    uint16_t num_cols;      // number of columns
    uint16_t num_data;      // length of ri,rx
    uint16_t reductiona;    // reduction offset in bp_tmpH
    uint16_t reductionlen;   // reduction length off bp_tmpH
} Collist;


void permute(int core_id);


void permuteT(int core_id);


void diaginv_lsolve(Diaginv* s);
/* 
 * Solves: x = mat*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void diaginv_ltsolve(Diaginv* s);
/* 
 * Solves: x = mat^T*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */

void diag_inv_mult(int core_id);
/* vector vector multiplication
 *   bp[] = bp[]*Dinv[]
 *   clear out data
 */


void collist_lsolve(Collist* s, int core_id);
/* 
 * Solves: Cx = bp
 *         and stores x in bp_tmp_h
 * Synchr: synchronizes cores
 * Copies: reduces bp_tmp_h0 up to bp_tmp_h7
 */

void collist_ltsolve(Collist* s);
/* 
 * Solves: C^Tx = bp
 */

#endif
