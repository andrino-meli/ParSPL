/*
 * Copyright 2024 ETH Zurich
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * Author: Andrino Meli (moroa@ethz.ch)
 */

#include <stdint.h>

#include "runtime.h"   // for runtime barrier
#include "types.h"
#include "parspl.h"   // for lsolve, ltsolve, solve
#include "workspace.h" // for access to x, GOLD, GOLD_INV
#include "verify.h"

#ifdef __RT_SSSR_ENABLE
#define SSSR
#endif


int smain(uint32_t core_id) {
// for verification purposes have different solve stages.
    __RT_SEPERATOR //for clean measurement have it outside.
#ifdef SSSR
    if (core_id < N_CCS) {
        asm volatile(
            __RT_SSSR_SCFGWI(%[icfg], 31,     __RT_SSSR_REG_IDX_CFG)
            __RT_SSSR_SCFGWI(%[stride], 31,   __RT_SSSR_REG_STRIDE_0)
            :: [stride]"r"(8), [icfg]"r"(__RT_SSSR_IDXSIZE_U16)
            : "memory"
        );
    }
#endif

// Run the linear system solver of choice
#ifdef PARSPL
    permute(core_id);
    // code generation determines wether ro run Lsolve, Ltsolve or both
    solve(core_id);
    permuteT(core_id);
    __RT_SEPERATOR
#elif defined SOLVE_CSC
    if (core_id == 0) {
        solve_csc();
    } else {
        __rt_get_timer();
        __rt_get_timer();
    }
    __RT_SEPERATOR
#elif defined PSOLVE_CSC
    psolve_csc(core_id);
    __RT_SEPERATOR
#elif defined SSSR_PSOLVE_CSC
    #ifdef SOLVE
    sssr_psolve_csc(core_id);
    #elif defined LSOLVE
    SSSR_PQDLDL_Lsolve(core_id);
    #elif defined LTSOLVE
    SSSR_PQDLDL_Ltsolve(core_id);
    #endif
    __RT_SEPERATOR
    // copy over result as sssr_Psolve_csc solves inplace in b and not into x
    if (core_id == 0) { for(int i = 0; i < LINSYS_N; i++) x[i] = b[i]; }
#else
    #error no solution method for the linear system is specified at preprocessing
#endif
    // Verify the result
    if (core_id == 0) {
        asm volatile("_check_for_nan_or_inf: \n":::);
        int r = is_normal_vec(x,192);
        if (!r) {
            //printf("Error: b is not normal after SSSR Lsolve!\n");
            return 7777;
        }
        return verify();
    }
    return 4242; // all other cores return 4242
}
