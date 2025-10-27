#ifndef GSARCHPackDef_H
#define GSARCHPackDef_H

#include <boost/preprocessor/list/for_each.hpp>
#include <ac_std_float.h>
#include <systemc.h>
#include <nvhls_module.h>
#include <nvhls_connections.h>
#include <mc_connections.h>
#include <ac_float.h>
#include <ac_fixed.h>
#include <nvhls_marshaller.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#define USE_FLOAT

#include "auto_gen_fields.h"

/*** Sorting / Tile vector types ***/
#define SORT_NUM 16
typedef ac_fixed<16, 6, false> SORT_ELEM_TYPE; // unsigned fixed-point key (16 total, 6 integer bits)
typedef ac_int<nvhls::log2_ceil<SORT_NUM>::val, false> log_sort_num;
typedef log_sort_num SORT_ID_TYPE; // minimize payload width to index size

#ifndef SORT_ASCENDING
#define SORT_ASCENDING 1 // 1: ascending (front-to-back), 0: descending
#endif

// Neutral vector type for sorting pipeline (key/payload in parallel arrays)
class SortingVec : public nvhls_message {
public:
    SORT_ELEM_TYPE key[SORT_NUM];   // e.g., depth/visibility metric per item
    SORT_ID_TYPE   id[SORT_NUM];    // payload/id aligned with key
    AUTO_GEN_FIELD_METHODS((key,id))
};

/***** Tile Merging specific types *****/
// Tile list length (use same as SORT_NUM for simplicity)
#define TM_LIST_LEN SORT_NUM
// Hot threshold (frequency >= TM_HOT_THRESH => hot)
#define TM_HOT_THRESH 2
// Hot buffer configuration (adjust to target macro mapping)
#define TM_NUM_BANKS 1
#define TM_NUM_ENTRIES 256

// Input list of Gaussian indices for a tile
class TM_IndexVec : public nvhls_message {
public:
    SORT_ID_TYPE id[TM_LIST_LEN];
    AUTO_GEN_FIELD_METHODS((id))
};

// Optional helper entry type (avoid using arrays of this in IO)
class TM_HotEntry : public nvhls_message {
public:
    SORT_ID_TYPE id;
    ac_int<16, false> addr;
    AUTO_GEN_FIELD_METHODS((id,addr))
};

// Output of TileMerging: hot entries with addresses + counts (parallel arrays to satisfy IO)
class TM_HCOut : public nvhls_message {
public:
    ac_int<8, false> hot_count;
    SORT_ID_TYPE     hot_id[TM_LIST_LEN];
    ac_int<16,false> hot_addr[TM_LIST_LEN];
    ac_int<8, false> cold_count;
    SORT_ID_TYPE     cold[TM_LIST_LEN];
    AUTO_GEN_FIELD_METHODS((hot_count,hot_id,hot_addr,cold_count,cold))
};

// Simple SRAM message and wrapper, patterned after GBR_Model style
#include <mem_array.h>

template <typename DataType>
class TM_sram_msg : public nvhls_message {
public:
    DataType data;
    NVUINT1 RW;        // 1 = write, 0 = read
    ac_int<16, false> addr;
    AUTO_GEN_FIELD_METHODS((data,RW,addr))
};

template <typename DataType, int NUMENTRIES, int NUMBANKS>
class TM_SRAM : public match::Module {
    SC_HAS_PROCESS(TM_SRAM);
public:
    Connections::In<TM_sram_msg<DataType>>  MemReqIn;
    Connections::Out<TM_sram_msg<DataType>> MemReqOut;

    mem_array_sep<DataType, (NUMENTRIES*NUMBANKS), NUMBANKS, 1> SRAM;

    TM_SRAM(sc_module_name name) : match::Module(name),
                                   MemReqIn("MemReqIn"),
                                   MemReqOut("MemReqOut") {
        SC_THREAD(RUN);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void RUN() {
        MemReqIn.Reset();
        MemReqOut.Reset();
        SRAM.clear();

        NVUINT1 have_req = NVUINT1(0);
        NVUINT1 is_write = NVUINT1(0);
        ac_int<16, false> addr_r = 0;
        DataType data_r = DataType();
        NVUINT1 addr_stage = NVUINT1(0);
        NVUINT1 resp_valid_r = NVUINT1(0);
        TM_sram_msg<DataType> resp_stage;

        wait();
        while (1) {
            wait();
            if (have_req == NVUINT1(0)) {
                TM_sram_msg<DataType> req;
                if (MemReqIn.PopNB(req)) {
                    have_req = NVUINT1(1);
                    is_write = req.RW;
                    addr_r   = req.addr;
                    data_r   = req.data;
                    addr_stage = NVUINT1(1);
                }
            }

            if (addr_stage == NVUINT1(1)) {
                ac_int<32, false> a = addr_r;
                ac_int<32, false> bank = a % NUMBANKS;
                ac_int<32, false> idx  = a / NUMBANKS;
                if (is_write == NVUINT1(1)) {
                    SRAM.write((int)idx, (int)bank, data_r);
                } else {
                    DataType rd = SRAM.read((int)idx, (int)bank);
                    resp_stage.data = rd;
                    resp_stage.RW   = NVUINT1(0);
                    resp_stage.addr = addr_r;
                    resp_valid_r = NVUINT1(1);
                }
                have_req = NVUINT1(0);
                addr_stage = NVUINT1(0);
            }

            if (resp_valid_r == NVUINT1(1)) {
                MemReqOut.Push(resp_stage);
                resp_valid_r = NVUINT1(0);
            }
        }
    }
};

/*** VRU-like types for simple feature/grad units ***/
typedef ac_std_float<16, 5> FP16_TYPE;
typedef ac_int<16, false> UINT16_TYPE;

class RGB_TYPE : public nvhls_message{
public:
    FP16_TYPE r;
    FP16_TYPE g;
    FP16_TYPE b;
    AUTO_GEN_FIELD_METHODS((r,g,b))
};

class VRU_IN_TYPE : public nvhls_message {
public:
    FP16_TYPE pixel_pos_x;
    FP16_TYPE pixel_pos_y;
    FP16_TYPE mean_x;
    FP16_TYPE mean_y;
    FP16_TYPE conx;
    FP16_TYPE cony;
    FP16_TYPE conz;
    RGB_TYPE   color;
    FP16_TYPE  opacity;
    FP16_TYPE  grad;        // scalar gradient G_i used by GradientPruning
    bool       last_gaussian;
    AUTO_GEN_FIELD_METHODS((pixel_pos_x,pixel_pos_y,mean_x,mean_y,conx,cony,conz,color,opacity,grad,last_gaussian))
};

class VRU_OUT_TYPE : public nvhls_message {
public:
    RGB_TYPE color;
    AUTO_GEN_FIELD_METHODS((color))
};

/*** Rearrangement Unit (RU) types ***/
#ifndef RU_NUM_BANKS
#define RU_NUM_BANKS 4
#endif

class RU_Req : public nvhls_message {
public:
    ac_int<16,false> addr; // flat address into Gradient Buffer
    FP16_TYPE        grad; // scalar gradient to accumulate
    bool             last; // tile boundary
    AUTO_GEN_FIELD_METHODS((addr,grad,last))
};

class RU_Update : public nvhls_message {
public:
    ac_int<nvhls::log2_ceil<RU_NUM_BANKS>::val,false> bank; // bank id
    ac_int<16,false> addr;  // address within GB (flat; banking resolved by bank field)
    FP16_TYPE        grad;  // gradient value
    bool             last;  // last update of this tile
    AUTO_GEN_FIELD_METHODS((bank,addr,grad,last))
};

#endif // GSARCHPackDef_H


