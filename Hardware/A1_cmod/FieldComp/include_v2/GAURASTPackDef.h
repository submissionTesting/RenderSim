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







/*** VRU Types ***/
#define NUM_ROTATE 4
typedef ac_int<nvhls::log2_ceil<NUM_ROTATE>::val, false> ROTATE_INDEX_TYPE; // 2-bit unsigned int for rotate index
// RGB color type
typedef ac_std_float<32, 8> FP32_TYPE;

class RGB_TYPE : public nvhls_message{
public:
    FP32_TYPE r;
    FP32_TYPE g;
    FP32_TYPE b;
    
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

// Input type for VRU
class VRU_IN_TYPE : public nvhls_message {
public:
    // Pixel position
    FP32_TYPE pixel_pos_x;
    FP32_TYPE pixel_pos_y;
    
    // Gaussian 2D mean (projected from 3D)
    FP32_TYPE mean_x;
    FP32_TYPE mean_y;
    
    // Gaussian 2D covariance matrix (projected from 3D)
    FP32_TYPE conx;
    FP32_TYPE cony;
    FP32_TYPE conz;
    
    // Gaussian color and opacity
    RGB_TYPE color;
    FP32_TYPE opacity;
    
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







/*** RE Constants ***/
/*** RE Types ***/

//                  16-bit, 8-bit precision
typedef ac_std_float<32, 8> POS_DATA_TYPE;
typedef ac_std_float<32, 8> COV2D_DATA_TYPE;
typedef ac_std_float<32, 8> OPACITY_DATA_TYPE;
typedef ac_std_float<32, 8> OUT_DATA_TYPE;
typedef ac_std_float<32, 8> COLOR_DATA_TYPE;
typedef ac_std_float<32, 8> alpha_type;
typedef ac_std_float<32, 8> PIXEL_DATA_TYPE;
const int NUM_PEs = 16;






// Pixel position (2D)
class PIXEL_POS_IN_TYPE : public nvhls_message {
public:
    POS_DATA_TYPE x[2];
    AUTO_GEN_FIELD_METHODS((x))
};

// Gaussian feature input (mean, covariance, opacity, color)
class GAUSS_FEATURE_IN_TYPE : public nvhls_message {
public:
    POS_DATA_TYPE x[2];            // mean
    COV2D_DATA_TYPE cov[4];        // 2x2 covariance matrix
    OPACITY_DATA_TYPE o;           // opacity
    COLOR_DATA_TYPE color[3];      // RGB color
    AUTO_GEN_FIELD_METHODS((x, cov, o, color))
};

// Output type
class OUT_TYPE : public nvhls_message {
public:
    OUT_DATA_TYPE x[4];
    AUTO_GEN_FIELD_METHODS((x))
};


class ARRAY_IN_TYPE : public nvhls_message {
public:
    PIXEL_POS_IN_TYPE x[NUM_PEs];
    AUTO_GEN_FIELD_METHODS((x))
};

class ARRAY_OUT_TYPE : public nvhls_message {
public:
    OUT_TYPE x[NUM_PEs];
    AUTO_GEN_FIELD_METHODS((x))
};


// Alpha intermediate type
class alpha_TYPE : public nvhls_message {
public:
    alpha_type x[4];
    AUTO_GEN_FIELD_METHODS((x))
};
class PE_CTRL_TYPE : public nvhls_message {
    public:
        bool EnableIn;     // Input enable
        bool EnableOut;     // Output enable
       
    
        PE_CTRL_TYPE() : EnableIn(false), EnableOut(false) {}
    
        AUTO_GEN_FIELD_METHODS((EnableIn, EnableOut))
};
#endif // ICARUSPackDef_H
