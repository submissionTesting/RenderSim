#ifndef SRENDERPackDef_H
#define SRENDERPackDef_H

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

// Max sizes for fixed-size arrays
#define MAX_RAYS 16
#define MAX_POINTS_PER_RAY 8
#define MAX_SENSITIVE_RAYS 16
#define MAX_SENSITIVE_POINTS 32
/*** SRender Type Definitions ***/
typedef ac_int<16, true> FP16_TYPE;    // 16-bit floating point
typedef ac_int<16, false> UINT16_TYPE;    // 16-bit unsigned int
typedef ac_int<8, false> UINT8_TYPE;      // 8-bit unsigned int

// RGB color type
class RGB_TYPE : public nvhls_message {
public:
    FP16_TYPE r;
    FP16_TYPE g;
    FP16_TYPE b;
    
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

/*** DCU (Distance Compute Unit) Types ***/
// DCU Input - provides two pixels to compare
class DCU_IN_TYPE : public nvhls_message {
public:
    RGB_TYPE informative_pixel;  // RGB values of the informative pixel
    RGB_TYPE candidate_pixel;    // RGB values of the candidate pixel
    
    AUTO_GEN_FIELD_METHODS((informative_pixel,candidate_pixel))
};

// DCU Output - contains L1 distance result
class DCU_OUT_TYPE : public nvhls_message {
public:
    FP16_TYPE distance;  // L1 distance between pixels
    
    AUTO_GEN_FIELD_METHODS((distance))
};

/*** CU (Comparison Unit) Types ***/
// CU Input - provides value and threshold to compare
class CU_IN_TYPE : public nvhls_message {
public:
    FP16_TYPE input_value;  // Value to compare (L1 distance or weight factor)
    FP16_TYPE threshold;    // Threshold value (T for ray sensitivity, D for point sensitivity)
    UINT16_TYPE id;         // ID of the ray or point being compared
    
    AUTO_GEN_FIELD_METHODS((input_value,threshold,id))
};

// CU Output - contains comparison result
class CU_OUT_TYPE : public nvhls_message {
public:
    UINT8_TYPE result;   // Comparison result (0 or 1)
    UINT16_TYPE id;      // ID of the ray or point (passed through)
    UINT16_TYPE cnt;
    
    AUTO_GEN_FIELD_METHODS((result,id,cnt))
};

// Structure for sensitive point information
class SensitivePointInfo : public nvhls_message {
public:
    UINT16_TYPE ray_id;           // ID of the sensitive ray
    UINT16_TYPE start_point_id;   // Start boundary point ID
    UINT16_TYPE end_point_id;     // End boundary point ID
    
    AUTO_GEN_FIELD_METHODS((ray_id,start_point_id,end_point_id))
};

// SPE input type
class SPE_IN_TYPE : public nvhls_message {
public:
    RGB_TYPE rendered_pixels[MAX_RAYS];     // Array of rendered pixel colors
    FP16_TYPE weight_factors[MAX_RAYS * MAX_POINTS_PER_RAY];     // Array of weight factors (Ti*αi)
    UINT16_TYPE num_rays;          // Number of rays
    UINT16_TYPE num_points_per_ray; // Number of points per ray
    FP16_TYPE threshold_T;         // Threshold for ray sensitivity
    FP16_TYPE threshold_D;         // Threshold for point sensitivity
    
    AUTO_GEN_FIELD_METHODS((rendered_pixels,weight_factors,num_rays,num_points_per_ray,threshold_T,threshold_D))
};

// SPE output type
class SPE_OUT_TYPE : public nvhls_message {
public:
    UINT16_TYPE sensitive_rays[MAX_SENSITIVE_RAYS];           // Array of sensitive ray IDs
    UINT16_TYPE num_sensitive_rays;        // Number of sensitive rays
    SensitivePointInfo sensitive_points[MAX_SENSITIVE_POINTS];  // Array of sensitive point information
    UINT16_TYPE num_sensitive_points;      // Number of sensitive point entries
    
    AUTO_GEN_FIELD_METHODS((sensitive_rays,num_sensitive_rays,sensitive_points,num_sensitive_points))
};


#define NPU_SIZE 32

typedef ac_int<4, true> NPU_W_Elem_Type;
typedef ac_int<4, true> NPU_In_Elem_Type;
typedef ac_int<4, true> NPU_Out_Elem_Type;

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
#endif //SRENDERPackDef_H
