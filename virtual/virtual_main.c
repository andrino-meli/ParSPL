// ----- Hardware virtualization -----
// Virtual runtime using pthread.
// Used to functionally verify code generation.

#include <pthread.h>
#include "runtime.h"

pthread_barrier_t barr_all;

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
    pthread_barrier_init(&barr_all, NULL, N_CCS);
    pthread_t cc[N_CCS];
    cc_params_t cc_params[N_CCS];
    // start threads
    for (int i=0; i<N_CCS; ++i) {
        cc_params[i].coreid = i;
        cc_params[i].num_cores = N_CCS;
        if (pthread_create(&cc[i], NULL, core_wrap, &cc_params[i])) return -i;
    }
    // join threads
    for (int i=0; i<N_CCS; ++i) {
        if (pthread_join(cc[i], NULL)) return -i;
    }
    // print return values:
    for (int i=0; i<N_CCS; ++i) {
        printf("Thread %d returned %d\n",i,cc_params[i].ret);
    }
    return 0;
}
