#ifndef GSARCHPackDef_H
#define GSARCHPackDef_H

#include <systemc.h>
#include <nvhls_module.h>
#include <ac_fixed.h>
#include <ac_int.h>
#include <ac_std_float.h>
#include <nvhls_marshaller.h>

#include "auto_gen_fields.h"

// Use FP16
using FP16_TYPE = ac_std_float<16,5>;
using F_mu    = FP16_TYPE;
using F_rot   = FP16_TYPE;
using F_K     = FP16_TYPE;
using F_z     = FP16_TYPE;
using F_invz  = FP16_TYPE;
using F_covar = FP16_TYPE;
using F_cov2d = FP16_TYPE;
using F_J     = FP16_TYPE;
using F_acc   = FP16_TYPE;
using F_cmp   = FP16_TYPE;
// using F_mu    = ac_fixed<16,  6, true,  AC_RND, AC_SAT>;   // Q6.10
// using F_rot   = ac_fixed<16,  6, true,  AC_RND, AC_SAT>;   // Q6.10
// using F_K     = ac_fixed<24, 16, true,  AC_RND, AC_SAT>;   // Q16.8
// using F_z     = ac_fixed<42, 34, false, AC_RND, AC_SAT>;   // depth
// using F_invz  = ac_fixed<34, 10, true,  AC_RND, AC_SAT>;   // Q10.24
// using F_covar = ac_fixed<24,  1, true,  AC_RND, AC_SAT>;   // Q1.23
// using F_cov2d = ac_fixed<40, 18, true,  AC_RND, AC_SAT>;   // Q20.24
// using F_J     = ac_fixed<40, 20, true,  AC_RND, AC_SAT>;
// using F_acc   = ac_fixed<40, 24, true,  AC_RND, AC_SAT>;   // Q12.28
// using F_cmp   = ac_fixed<50, 36, true,  AC_RND, AC_SAT>;


static const int  SCREEN_WIDTH  = 800;
static const int  SCREEN_HEIGHT = 800;

class float3 : public nvhls_message {
public:
  F_mu x, y, z;
  AUTO_GEN_FIELD_METHODS((x,y,z))
};

class float2 : public nvhls_message {
public:
  F_K x, y;
  AUTO_GEN_FIELD_METHODS((x,y))
};

class CameraParams : public nvhls_message {
public:
  F_rot view[4][4];
  F_K   K[3][3];
  F_z   near_z;
  F_z   far_z;
  AUTO_GEN_FIELD_METHODS((view,K,near_z,far_z))
};

class Stage1a_In : public nvhls_message {
public:
  float3 mu_w;
  F_covar covmat[3][3];
  ac_int<32,false> ID;
  bool finish;
  AUTO_GEN_FIELD_METHODS((mu_w,covmat,ID,finish))
};

class Stage1f_Out : public nvhls_message {
public:
  float2 uv;
  ac_int<32,false> ID;
  F_cov2d cov2d[2][2];
  F_K radius_px;
  bool finish;
  AUTO_GEN_FIELD_METHODS((uv,ID,cov2d,radius_px,finish))
};

#endif // GSARCHPackDef_H


