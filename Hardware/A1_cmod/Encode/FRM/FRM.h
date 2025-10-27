#ifndef INSTANT3D_FRM_H
#define INSTANT3D_FRM_H

#include "INSTANT3DPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class FRM : public match::Module {
    SC_HAS_PROCESS(FRM);
public:
    Connections::In<FRM_IN_TYPE> FRMInput;
    Connections::Out<FRM_OUT_TYPE> FRMOutput;

    FRM(sc_module_name name) : match::Module(name),
                               FRMInput("FRMInput"),
                               FRMOutput("FRMOutput") {
        SC_THREAD(FRM_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void FRM_CALC() {
        FRMInput.Reset();
        FRMOutput.Reset();
        wait();

        // Feed-Forward Read Mapper: reorder requests to maximize parallel bank utilization
        static const int NUM_BANKS = 8;
        // simple buffers per bank for one timestep window
        FP16_TYPE bank_buf[NUM_BANKS];
        bool      bank_valid[NUM_BANKS];
        for (int b = 0; b < NUM_BANKS; b++) { 
            bank_valid[b]=false; 
            bank_buf[b]=FP16_TYPE(0.0); 
        }

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            FRM_IN_TYPE in;
            if (FRMInput.PopNB(in)) {
                // Addr Generator: hash to bank id using address LSBs
                ac_int<16,false> addr = in.addr;
                int bank = (int)(addr & (NUM_BANKS-1));

                // Bank Collision Detector: if bank already taken this timestep, defer (keep previous)
                if (!bank_valid[bank]) {
                    bank_buf[bank] = in.f; bank_valid[bank] = true;
                }
            }

            // Read Commit Unit: emit compacted high-utilization sequence (banks 0..7)
            for (int b = 0; b < NUM_BANKS; b++) {
                if (bank_valid[b]){
                    FRM_OUT_TYPE o; o.g = bank_buf[b]; o.addr = (ac_int<16,false>)b; FRMOutput.PushNB(o);
                    bank_valid[b]=false;
                }
            }
        }
    }
};

#endif // INSTANT3D_FRM_H


