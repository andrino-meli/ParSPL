#include <stdint.h>


void permute(int core_id);

void permuteT(int core_id);

void diaginv_lsolve(
/* 
 * Solves: x = mat*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */
    unsigned int n, // shape(mat) = (n,n)
    unsigned int rowa, // row/column offset
    double mat[], // invrse matrix
    uint16_t assigned_rows[],
    unsigned int num_rows // len(assigned_rows)
);

void diaginv_ltsolve(
/* 
 * Solves: x = mat^T*bp
 *         and stores x in bp_cp
 * Synchr: synchronizes cores
 * Copies: relevent part of bp_cp to bp
 */
    unsigned int n, // shape(mat) = (n,n)
    unsigned int rowa, // row/column offset
    double mat[], // invrse matrix
    uint16_t assigned_rows[],
    unsigned int num_rows // len(assigned_rows)
);

void diag_inv_mult(
/* vector vector multiplication
 *   bp[] = bp[]*Dinv[]
 * TODO: use linalg files for this
 */
    int core_id
);


void collist_lsolve(
/* 
 * Solves: Cx = bp
 *         and stores x in bp_tmp
 * Synchr: synchronizes cores
 * Copies: reduces bp_h0 up to bp_h10 into bp
 */
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
);

void collist_ltsolve(
/* 
 * Solves: C^Tx = bp
 */
    uint16_t num_cols, // number of columns
    uint16_t assigned_cols[], // list of assigned columns
    uint16_t len_cols[], // length of each column
    uint16_t num_data, // length of ri,rx
    uint16_t ri[], // row index
    double rx[] // row data
);
