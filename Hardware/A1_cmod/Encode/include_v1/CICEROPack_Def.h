

#ifndef encode_Def_H
#define encode_Def_H

#include <boost/preprocessor/list/for_each.hpp>
#include <ac_std_float.h>
#include <systemc.h>
#include <nvhls_module.h>
#include <ac_float.h>
#include <ac_fixed.h>
#include <ac_int.h>
#include <nvhls_marshaller.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
//#define USE_FLOAT

// Helpers from https://example.com/matchlib_toolkit/auto_gen_fields.h#L425
// Slightly modify for marshall, width only
#include "auto_gen_fields.h"

#define NPU_SIZE 24


#ifdef USE_FLOAT
typedef ac_std_float<16, 8> TESTTYPE;
#else
//typedef ac_fixed<32, 16, true, AC_TRN, AC_SAT> TESTTYPE; // saturated to nearest max/min
typedef ac_fixed<9, 9, true, AC_TRN, AC_SAT> TESTTYPE; // saturated to nearest max/min
#endif


typedef ac_int<16, true> NPU_W_Elem_Type;
typedef ac_int<16, true> NPU_In_Elem_Type;
typedef ac_int<16, true> NPU_Out_Elem_Type;
typedef ac_int<nvhls::log2_ceil<NPU_SIZE>::val+1, true> NPU_Index_Type;  // Generic index type for both row and column


// MUL types
static int const MUL_PART     = 2;                    // number of selectors
static int const BIT_PER_PART = 4;                    // selector width
static int const INTEGER_WIDTH = 8;                   // fixed point integer width
static int const total_bit = MUL_PART*BIT_PER_PART+1; // +1 => sign bit
typedef ac_fixed<total_bit, INTEGER_WIDTH+1, true, AC_TRN, AC_SAT> MUL_In_Type;  // changeable
typedef ac_fixed<total_bit, INTEGER_WIDTH+1, true, AC_TRN, AC_SAT> MUL_Out_Type; // changeable
#define SCALE 128 // If use actual fixed-point (INTEGER_WIDTH < tota_bit-1), then it is 1 (since handled in fixed)

class NPU_W_Type : public nvhls_message {
public:
    NPU_W_Elem_Type X[NPU_SIZE];
    AUTO_GEN_FIELD_METHODS((X))
};

class NPU_In_Type : public nvhls_message {
public:
    NPU_In_Elem_Type X[NPU_SIZE];
    AUTO_GEN_FIELD_METHODS((X))
};

class NPU_Out_Type : public nvhls_message {
public:
    NPU_Out_Elem_Type X[NPU_SIZE];
    AUTO_GEN_FIELD_METHODS((X))
};

// Address Generation
typedef ac_int<32, false> AG_VID;
typedef ac_int<16, false> AG_W;
class AG_VIDsWs : public nvhls_message { // 48B
public:
    AG_VID v[8];
    AG_W   w[8];
    AUTO_GEN_FIELD_METHODS((v, w))
};

// Reducer
typedef ac_int<16, false> reducer_W;
typedef ac_int<16, false> reducer_feature;

#endif //CICEROPackDef_H
