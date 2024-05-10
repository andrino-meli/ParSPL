#include <stdint.h>

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
