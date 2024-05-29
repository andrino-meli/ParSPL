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
 * Author: Andrino Meli (adinolp.xp@gmail.com, moroa@ethz.ch)
 */

#include <stdint.h>

#include "runtime.h"   // for runtime barrier
#include "parspl.h"   // for lsolve, ltsolve, solve
#include "kernel.h"   // for permute, permuteT
#include "workspace.h" // for access to x, GOLD, GOLD_INV
#include "print_float.h"

#define TOL (1e2) // require the error to be less than 1%

void __rt_seperator() {
    __rt_fpu_fence_full();
    __rt_barrier();
    __rt_get_timer();
}

int verify() {
    double maxerr = 0;
    double maxrelerr = 0;

    for(int j = 0; j < LINSYS_N; j++){
        // Access x in permutated order to be alligns with the plots.
        int i = Perm[j];
        // absolute error
        double err = (double)x[i] - XGOLD[i];
        double abserr = ( err >= 0 ) ? err : -err;
        if (maxerr < abserr) maxerr = abserr;
        // relative error
        double relerr = err*XGOLD_INV[i];
        double absrel = ( relerr >= 0 ) ? relerr : -relerr;
        #ifdef VERBOSE
        printf("prow %d:\t\tgold %.3e, x %.3e,\terr %.3e,\tabsrel %.3f%%\trow %d\n",
                j, XGOLD[i], x[i], err, absrel*100, i);
        #elif PRINTF
        printf("prow %d:\t\tx ",j);
        printFloat(x[i]);
        printf("\n");
        #endif
        if (maxrelerr < absrel) {
            maxrelerr = absrel;
            #ifdef VERBOSE
            printf("Updating maxrelerr on prow %d\n",i);
            #endif
        }
    }
    #ifdef PRINTF
        #ifdef LSOLVE
        printf("\nVERIFICATION of lsolve:\n");
        #endif
        #ifdef LTSOLVE
        printf("\nVERIFICATION of ltsolve:\n");
        #endif
        #ifdef SOLVE
        printf("\nVERIFICATION of ldlsolve:\n");
        #endif
    printf(" Maximum absolute error %e\n", maxrelerr);
    printf(" Maximum relative error %e\n", maxerr);
    printf(" Thread 0 will return max rel err in %%\n");
    #endif
    return (int)(maxrelerr*TOL);
}



int smain(uint32_t core_id, uint32_t core_num) {
// for verification purposes have different solve stages.
    __rt_seperator(); // for measuring
#ifdef LSOLVE
    permute(core_id);
    lsolve(core_id);
    permuteT(core_id);
#endif
#ifdef LTSOLVE
    permute(core_id);
    ltsolve(core_id);
    permuteT(core_id);
#endif
#ifdef SOLVE
    permute(core_id);
    solve(core_id);
    permuteT(core_id);
#endif
    if (core_id == 0) {
        return verify();
    }
    // all other cores return 4242
    return 4242;
}
