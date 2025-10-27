#ifndef INSTANT3DPackDef_H
#define INSTANT3DPackDef_H

#include <boost/preprocessor/list/for_each.hpp>
#include <ac_std_float.h>
#include <systemc.h>
#include <nvhls_module.h>
#include <ac_float.h>
#include <ac_fixed.h>
#include <nvhls_marshaller.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#define USE_FLOAT

#include "auto_gen_fields.h"

typedef ac_std_float<16, 5> FP16_TYPE;
typedef ac_int<16, false> UINT16_TYPE;

class FRM_IN_TYPE : public nvhls_message {
public:
    FP16_TYPE f;              // payload (optional)
    ac_int<16,false> addr;    // request address (synth-friendly)
    AUTO_GEN_FIELD_METHODS((f,addr))
};

class FRM_OUT_TYPE : public nvhls_message {
public:
    FP16_TYPE g;              // payload (optional)
    ac_int<16,false> addr;    // committed address
    AUTO_GEN_FIELD_METHODS((g,addr))
};

class BUM_IN_TYPE : public nvhls_message {
public:
    ac_int<16,false> addr;   // update address (index)
    FP16_TYPE        grad;   // input gradient
    bool             last;   // flush marker for current window
    AUTO_GEN_FIELD_METHODS((addr,grad,last))
};

class BUM_OUT_TYPE : public nvhls_message {
public:
    ac_int<16,false> addr;   // address to write back
    FP16_TYPE        upd;    // merged update (after learning-rate scale)
    AUTO_GEN_FIELD_METHODS((addr,upd))
};

#endif // INSTANT3DPackDef_H


