//
// HybridDef.h
// Centralized ML data type definitions and common NVHLS/Matchlib includes
// Used by NN_Model, SVR_Model, KNN_Model, GBR_Model
//

#ifndef HYBRID_DEF_H
#define HYBRID_DEF_H

// Consolidated core includes needed by all ML model headers.
// Bring in SystemC and Matchlib/NVHLS marshalling utilities and macros
#include <systemc.h>
#include <ac_std_float.h>
#include <ac_int.h>
#include <nvhls_connections.h>
#include <nvhls_module.h>
#include <nvhls_marshaller.h>
#include <nvhls_int.h>
#include <nvhls_types.h>

// AUTO_GEN_FIELD_METHODS and marshalling helpers used by message classes
#include "auto_gen_fields.h"

// Common ML scalar types
typedef ac_std_float<16, 8> ML_FLOAT;
typedef ac_std_float<16, 8> ML_WEIGHT;

#include <mem_array.h>

// Generic SRAM request/response message carrying DataType payload
template <typename DataType>
class sram_msg : public nvhls_message {
public:
    DataType data;
    NVUINT1 RW;        // 1 = write, 0 = read
    NVUINT16 addr;     // flat address; banking handled internally
    AUTO_GEN_FIELD_METHODS((data, RW, addr));
};

// Pipelined SRAM wrapper (register addr/CS one cycle before access) patterned after 1_NEUREX/SRAMTest
// NUMENTRIES: entries per bank; NUMBANKS: number of banks
template <typename DataType, int NUMENTRIES, int NUMBANKS>
class GBR_SRAM : public match::Module {
    SC_HAS_PROCESS(GBR_SRAM);
public:
    Connections::In<sram_msg<DataType>>  MemReqIn;
    Connections::Out<sram_msg<DataType>> MemReqOut;

    // mem_array_sep expects total entries across all banks as the 2nd template param
    mem_array_sep<DataType, (NUMENTRIES*NUMBANKS), NUMBANKS, 1> SRAM;

    GBR_SRAM(sc_module_name name) : match::Module(name),
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

        // pipeline registers
        NVUINT1 have_req = NVUINT1(0);
        NVUINT1 is_write = NVUINT1(0);
        NVUINT16 addr_r = NVUINT16(0);
        DataType data_r = DataType();
        NVUINT1 addr_stage = NVUINT1(0);
        // add an explicit response pipeline to model a 1-cycle read latency
        NVUINT1 resp_valid_r = NVUINT1(0);
        sram_msg<DataType> resp_stage;

        wait();
        while (1) {
            wait();

            // Stage 0: capture new request when idle
            if (have_req == NVUINT1(0)) {
                sram_msg<DataType> req;
                if (MemReqIn.PopNB(req)) {
                    have_req = NVUINT1(1);
                    is_write = req.RW;
                    addr_r   = req.addr;
                    data_r   = req.data;
                    addr_stage = NVUINT1(1); // schedule access next cycle
                }
            }

            // Stage 1: perform SRAM access with registered addr/control
            if (addr_stage == NVUINT1(1)) {
                ac_int<32, false> a = ac_int<32,false>(addr_r.to_uint());
                ac_int<32, false> bank = a % NUMBANKS;
                ac_int<32, false> idx  = a / NUMBANKS;
                if (is_write == NVUINT1(1)) {
                    SRAM.write((int)idx, (int)bank, data_r);
                    have_req = NVUINT1(0);
                } else {
                    DataType rd = SRAM.read((int)idx, (int)bank);
                    resp_stage.data = rd;
                    resp_stage.RW   = NVUINT1(0);
                    resp_stage.addr = addr_r;
                    resp_valid_r = NVUINT1(1); // will push next cycle
                    have_req = NVUINT1(0);
                }
                addr_stage = NVUINT1(0);
            }

            // Stage 2: return read data (one cycle after SRAM read)
            if (resp_valid_r == NVUINT1(1)) {
                MemReqOut.Push(resp_stage);
                resp_valid_r = NVUINT1(0);
            }
        }
    }
};

#endif // HYBRID_DEF_H


