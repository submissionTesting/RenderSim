#ifndef IRISPackDef_H
#define IRISPackDef_H

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

/*** RFM Constants ***/
#define SORT_NUM 16
/*** RFM Types ***/
//                  16-bit, 8-bit precision
typedef ac_std_float<16, 8> RFM_IN_OUT_TYPE;
typedef NVUINTW(248) RFM_MODE_BIT;





#endif //IRISPackDef_H
