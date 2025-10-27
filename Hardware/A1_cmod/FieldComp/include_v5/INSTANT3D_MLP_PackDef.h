#ifndef INSTANT3D_MLP_PackDef_H
#define INSTANT3D_MLP_PackDef_H

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

class MLP3D_IN_TYPE : public nvhls_message {
public:
    FP16_TYPE x;
    AUTO_GEN_FIELD_METHODS((x))
};

class MLP3D_OUT_TYPE : public nvhls_message {
public:
    FP16_TYPE y;
    AUTO_GEN_FIELD_METHODS((y))
};

#endif // INSTANT3D_MLP_PackDef_H


