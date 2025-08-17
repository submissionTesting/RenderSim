#ifndef ICARUSPackDef_H
#define ICARUSPackDef_H

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

// Helpers from https://example.com/matchlib_toolkit/auto_gen_fields.h#L425
// Slightly modify for marshall, width only
#include "auto_gen_fields.h"

typedef ac_int<8, true> INT8_TYPE;     // 16-bit unsigned int for GID
typedef ac_int<12, true> INT12_TYPE;     // 16-bit unsigned int for GID
typedef ac_std_float<16, 5> FP16_TYPE;     // 16-bit floating point for Depth
typedef ac_int<16, false> UINT16_TYPE;     // 16-bit unsigned int for GID

/*** VRU Types ***/
#define NUM_ROTATE 4
typedef ac_int<nvhls::log2_ceil<NUM_ROTATE>::val, false> ROTATE_INDEX_TYPE; // 2-bit unsigned int for rotate index
// RGB color type
class RGB_TYPE : public nvhls_message{
public:
    FP16_TYPE r;
    FP16_TYPE g;
    FP16_TYPE b;
    
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

class RGB_INT_TYPE : public nvhls_message{
public:
    INT8_TYPE r;
    INT8_TYPE g;
    INT8_TYPE b;
    
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

// Input type for VRU
class VRU_IN_TYPE : public nvhls_message {
public:
    // Pixel position
    FP16_TYPE pixel_pos_x;
    FP16_TYPE pixel_pos_y;
    
    // Gaussian 2D mean (projected from 3D)
    FP16_TYPE mean_x;
    FP16_TYPE mean_y;
    
    // Gaussian 2D covariance matrix (projected from 3D)
    FP16_TYPE conx;
    FP16_TYPE cony;
    FP16_TYPE conz;
    
    // Gaussian color and opacity
    RGB_TYPE color;
    FP16_TYPE opacity;
    
    // Bitmap for subtile skipping
    // UINT8_TYPE bitmap;
    // UINT8_TYPE subtile_idx;
    
    // Flag to indicate last Gaussian for this pixel
    bool last_gaussian;

    ROTATE_INDEX_TYPE rotate_idx;

    AUTO_GEN_FIELD_METHODS((pixel_pos_x,pixel_pos_y,mean_x,mean_y,
                           conx,cony,conz,
                           color,opacity,/*(bitmap)(subtile_idx)*/last_gaussian,rotate_idx))
};

// Input type for VRU
class VRU_IN_INT_TYPE : public nvhls_message {
public:
    // Pixel position
    INT8_TYPE pixel_pos_x;
    INT8_TYPE pixel_pos_y;
    
    // Gaussian 2D mean (projected from 3D)
    INT8_TYPE mean_x;
    INT8_TYPE mean_y;
    
    // Gaussian 2D covariance matrix (projected from 3D)
    INT8_TYPE conx;
    INT8_TYPE cony;
    INT8_TYPE conz;
    
    // Gaussian color and opacity
    RGB_INT_TYPE color;
    INT8_TYPE opacity;
    
    // Bitmap for subtile skipping
    // UINT8_TYPE bitmap;
    // UINT8_TYPE subtile_idx;
    
    // Flag to indicate last Gaussian for this pixel
    bool last_gaussian;

    ROTATE_INDEX_TYPE rotate_idx;

    AUTO_GEN_FIELD_METHODS((pixel_pos_x,pixel_pos_y,mean_x,mean_y,
                           conx,cony,conz,
                           color,opacity,/*(bitmap)(subtile_idx)*/last_gaussian,rotate_idx))
};

// Output type for VRU
class VRU_OUT_TYPE : public nvhls_message {
public:
    RGB_TYPE color;
    
    AUTO_GEN_FIELD_METHODS((color))
};

class VRU_OUT_INT_TYPE : public nvhls_message {
public:
    RGB_INT_TYPE color;
    
    AUTO_GEN_FIELD_METHODS((color))
};


#endif //ICARUSPackDef_H
