#ifndef INSTANT3D_BUM_H
#define INSTANT3D_BUM_H

#include "INSTANT3DPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class BUM : public match::Module {
    SC_HAS_PROCESS(BUM);
public:
    Connections::In<BUM_IN_TYPE> BUMInput;
    Connections::Out<BUM_OUT_TYPE> BUMOutput;

    BUM(sc_module_name name) : match::Module(name),
                               BUMInput("BUMInput"),
                               BUMOutput("BUMOutput") {
        SC_THREAD(BUM_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void BUM_CALC() {
        BUMInput.Reset();
        BUMOutput.Reset();
        wait();

        // Back-Propagation Update Merger (BUM)
        // One-to-all match over CAM-like entry table: merge updates to same address; create new if miss
        static const int NUM_ENTRIES = 16;
        ac_int<16,false> entry_addr[NUM_ENTRIES];
        FP16_TYPE        entry_val [NUM_ENTRIES];
        bool             entry_vld [NUM_ENTRIES];
        for (int i=0;i<NUM_ENTRIES;i++){ entry_vld[i]=false; entry_addr[i]=0; entry_val[i]=FP16_TYPE(0.0); }

        FP16_TYPE lr = FP16_TYPE(1.0); // learning rate scale placeholder

        bool commit_phase = false;
        int  commit_idx = 0;

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            BUM_IN_TYPE in;
            if (!commit_phase && BUMInput.PopNB(in)) {
                // One-to-all match
                int match_idx = -1; int free_idx = -1;
                for (int i=0;i<NUM_ENTRIES;i++){
                    if (entry_vld[i] && entry_addr[i]==in.addr) { match_idx = i; }
                    if (!entry_vld[i] && free_idx==-1) { free_idx = i; }
                }

                if (match_idx >= 0) {
                    // Merge: entry += grad
                    entry_val[match_idx] = entry_val[match_idx] + in.grad;
                } else if (free_idx >= 0) {
                    // Create new
                    entry_vld[free_idx] = true; entry_addr[free_idx] = in.addr; entry_val[free_idx] = in.grad;
                }

                // Start commit phase on last flag
                if (in.last) { commit_phase = true; commit_idx = 0; }
            }

            // Commit at most one entry per cycle to honor channel handshake
            if (commit_phase) {
                bool pushed = false;
                for (int i = commit_idx; i < NUM_ENTRIES && !pushed; i++) {
                    if (entry_vld[i]) {
                        BUM_OUT_TYPE o; o.addr = entry_addr[i]; o.upd = entry_val[i] * lr; BUMOutput.PushNB(o);
                        entry_vld[i] = false; entry_val[i] = FP16_TYPE(0.0);
                        commit_idx = i + 1;
                        pushed = true;
                    }
                }
                if (!pushed) {
                    // No more valid entries; end commit phase
                    commit_phase = false; commit_idx = 0;
                }
            }
        }
    }
};

#endif // INSTANT3D_BUM_H


