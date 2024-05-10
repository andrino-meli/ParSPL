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

#include "runtime.h"   // for access to barrier
#include "parspl.h"   // for lsolve, ltsolve, solve functions
#include "workspace.h" // for access to bp

#define TOL (1e2) // require the error to be less than 1%

int verify() {
    #ifdef VERBOSE
    printf("VERIFICATION:\n");
    #endif
    double maxerr = 0;
    double maxrelerr = 0;

    for(int i = 0; i < LINSYS_N; i++){
        // absolute error
        double err = (double)bp[i] - XGOLD[i];
        double abserr = ( err >= 0 ) ? err : -err;
        if (maxerr < abserr) maxerr = abserr;
        // relative error
        double relerr = err*XGOLD_INV[i];
        double absrel = ( relerr >= 0 ) ? relerr : -relerr;
        #ifdef VERBOSE
        printf("row %d:\t\tgold %.3e, bp %.3e,\terr %.3e,\tabsrel %.3f%%\n", i, XGOLD[i], bp[i], err, absrel*100);
        #endif
        if (maxrelerr < absrel) {
            maxrelerr = absrel;
            #ifdef VERBOSE
            printf("Updating maxrelerr on row %d\n",i);
            #endif
        }
    }
    #ifdef VERBOSE
    printf("maxrelerr %e\n", maxrelerr);
    #endif
    return (int)(maxrelerr*TOL);
}

int smain(uint32_t core_id, uint32_t core_num) {
// for verification purposes have different solve stages.
#ifdef LSOLVE
    lsolve(core_id);
#endif
#ifdef LTSOLVE
    ltsolve(core_id);
#endif
#ifdef SOLVE
    solve(core_id);
#endif
    __rt_barrier();
    if (core_id == 0) {
        return verify();
    }
    // all other cores return 4242
    return 4242;
}
