#undef FP32
#undef FP64

#define FP64

#ifdef FP32
    typedef float FLOAT;
    #define FMV_INS "fmv.s "
    #define FMADD_INS "fmadd.s "
    #define FADD_INS "fadd.s "
    #define FNMSUB_INS "fnmsub.s "
    #define FSUB_INS "fsub.s "
    #define FMUL_INS "fmul.s "
    #define FSW_INS "fsw "
    #define FLW_INS "flw "
#elif defined FP64
    typedef double FLOAT;
    #define FMV_INS "fmv.d "
    #define FMADD_INS "fmadd.d "
    #define FADD_INS "fadd.d "
    #define FNMSUB_INS "fnmsub.d "
    #define FSUB_INS "fsub.d "
    #define FMUL_INS "fmul.d "
    #define FSW_INS "fsd "
    #define FLW_INS "fld "
#else
    #warning no floating point format specified
#endif
