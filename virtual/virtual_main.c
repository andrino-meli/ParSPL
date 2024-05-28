// ----- Hardware virtualization -----
// Virtual runtime using pthread.
// Used to functionally verify code generation.

#include <pthread.h>
#include "runtime.h"

// ----- Runtime substitutes -----
pthread_barrier_t barr_all;
//spawn 3 additional core to check that the code handles core mismatch configuration
#define N_CCS_MISSMATCH (N_CCS + 3)

void __rt_barrier() {
    pthread_barrier_wait(&barr_all);
}

void __rt_fpu_fence_full() {
    return;
}

uint32_t __rt_get_timer() {
    return 0;
}

typedef struct {
    uint32_t coreid;
    uint32_t num_cores;
    int ret;
} cc_params_t;


void *core_wrap(void *params) {
    cc_params_t* cc_params = params;
    cc_params->ret = smain(cc_params->coreid, cc_params->num_cores);
    return NULL;
}


int main() {
    // create thread pool
    pthread_barrier_init(&barr_all, NULL, N_CCS_MISSMATCH);
    pthread_t cc[N_CCS_MISSMATCH];
    cc_params_t cc_params[N_CCS_MISSMATCH];
    // start threads
    for (int i=0; i<N_CCS_MISSMATCH; ++i) {
        cc_params[i].coreid = i;
        cc_params[i].num_cores = N_CCS_MISSMATCH;
        if (pthread_create(&cc[i], NULL, core_wrap, &cc_params[i])) return -i;
    }
    // join threads
    for (int i=0; i<N_CCS_MISSMATCH; ++i) {
        if (pthread_join(cc[i], NULL)) return -i;
    }
    // print return values:
    for (int i=0; i<N_CCS_MISSMATCH; ++i) {
        printf("Thread %d returned %d\n",i,cc_params[i].ret);
    }
    return 0;
}
