#ifndef GBUPackDef_H
#define GBUPackDef_H

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

class RGB16 : public nvhls_message {
public:
    FP16_TYPE r, g, b;
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

class ROW_IN_TYPE : public nvhls_message {
public:
    // Threshold compute inputs
    FP16_TYPE dxpp;      // Δx''
    FP16_TYPE xpp;       // x''
    FP16_TYPE yn2;       // y_n^2 (or y'')^2 equivalent
    FP16_TYPE thr;       // threshold
    // Color compute inputs
    FP16_TYPE opacity;   // alpha
    RGB16     color;     // per-sample color
    ac_int<16,false> pix_idx; // pixel position along row
    bool last;           // end-of-row marker
    AUTO_GEN_FIELD_METHODS((dxpp,xpp,yn2,thr,opacity,color,pix_idx,last))
};

class ROW_OUT_TYPE : public nvhls_message {
public:
    RGB16 accum;               // accumulated color for this pixel
    FP16_TYPE T;               // remaining transmittance after this pixel
    ac_int<16,false> pix_idx;  // pixel position
    bool last;                 // last pixel in row
    AUTO_GEN_FIELD_METHODS((accum,T,pix_idx,last))
};

// Config for RowGeneration LUT entries
class RG_LUT_CFG : public nvhls_message {
public:
    UINT16_TYPE idx;
    FP16_TYPE dxpp;
    FP16_TYPE xpp_base;
    FP16_TYPE yn2;
    FP16_TYPE thr;
    FP16_TYPE opacity;
    RGB16     color;
    AUTO_GEN_FIELD_METHODS((idx,dxpp,xpp_base,yn2,thr,opacity,color))
};

// Default LUT size parameter
#ifndef GBU_LUT_SIZE
#define GBU_LUT_SIZE 16
#endif

class BIN_DESC_TYPE : public nvhls_message {
public:
    UINT16_TYPE bin_idx;
    UINT16_TYPE count;
    AUTO_GEN_FIELD_METHODS((bin_idx,count))
};

#endif // GBUPackDef_H


