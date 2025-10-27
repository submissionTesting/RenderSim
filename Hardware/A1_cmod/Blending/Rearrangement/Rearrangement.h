#ifndef GSARCH_REARRANGEMENT_H
#define GSARCH_REARRANGEMENT_H

#include "../include_v6/GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class Rearrangement : public match::Module {
    SC_HAS_PROCESS(Rearrangement);
public:
    // Input: per-Gaussian gradient request carrying address and G_i
    Connections::In<RU_Req>   RearrangementInput;
    // Output: banked updates for accumulation units (no SRAM here)
    Connections::Out<RU_Update> RearrangementOutput;

    Rearrangement(sc_module_name name) : match::Module(name),
                                         RearrangementInput("RearrangementInput"),
                                         RearrangementOutput("RearrangementOutput") {
        SC_THREAD(run);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void run() {
        RearrangementInput.Reset();
        RearrangementOutput.Reset();
        wait();

        // RU: mask controller + accumulation selection without SRAM
        const int RU_POOL_SIZE = 8;
        RU_Req pool[RU_POOL_SIZE];
        bool   valid[RU_POOL_SIZE];
        for (int i = 0; i < RU_POOL_SIZE; i++) valid[i] = false;
        int pool_count = 0;

        RU_Update emit_buf[RU_NUM_BANKS];
        int emit_count = 0;
        int emit_idx = 0;
        bool emitting = false;
        bool last_seen = false;

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();

            // 1) Ingest new request into pool
            RU_Req in;
            if (RearrangementInput.PopNB(in)) {
                if (pool_count < RU_POOL_SIZE) {
                    int slot = RU_POOL_SIZE;
                    for (int i = 0; i < RU_POOL_SIZE; i++) {
                        bool is_free = (valid[i] == false);
                        slot = (slot == RU_POOL_SIZE && is_free) ? i : slot;
                    }
                    if (slot != RU_POOL_SIZE) {
                        pool[slot] = in; valid[slot] = true; pool_count++;
                        if (in.last) last_seen = true;
                    }
                }
            }

            // 2) If not currently emitting, select up to RU_NUM_BANKS non-conflicting banks
            if (!emitting && pool_count > 0) {
                unsigned bank_mask = 0u;
                emit_count = 0;
                for (int i = 0; i < RU_POOL_SIZE; i++) {
                    bool slot_valid = valid[i];
                    bool full = (emit_count >= RU_NUM_BANKS);
                    if (slot_valid && !full) {
                        int bank = (int)(pool[i].addr & (RU_NUM_BANKS-1));
                        bool bank_free = (((bank_mask >> bank) & 1u) == 0u);
                        if (bank_free) {
                            // Accumulate same bank+addr within this selection
                            ac_int<16,false> addr = pool[i].addr;
                            FP16_TYPE sum = pool[i].grad;
                            valid[i] = false; pool_count--;
                            for (int j = i+1; j < RU_POOL_SIZE; j++) {
                                bool j_valid = valid[j];
                                if (j_valid) {
                                    int bj = (int)(pool[j].addr & (RU_NUM_BANKS-1));
                                    bool same = (bj == bank) && (pool[j].addr == addr);
                                    if (same) {
                                        sum = sum + pool[j].grad;
                                        valid[j] = false; pool_count--;
                                    }
                                }
                            }
                            emit_buf[emit_count].bank = bank;
                            emit_buf[emit_count].addr = addr;
                            emit_buf[emit_count].grad = sum;
                            emit_buf[emit_count].last = false;
                            emit_count = emit_count + 1;
                            bank_mask |= (1u << bank);
                        }
                    }
                }
                emit_idx = 0; emitting = (emit_count > 0);
            }

            // 3) Emit one update per cycle, set last on the very final update of the tile
            if (emitting) {
                RU_Update u = emit_buf[emit_idx];
                bool is_last = (emit_idx == (emit_count - 1)) && (pool_count == 0) && last_seen;
                if (is_last) { u.last = true; last_seen = false; } else { u.last = false; }
                RearrangementOutput.PushNB(u);
                emit_idx++;
                if (emit_idx >= emit_count) { emitting = false; }
            }
        }
    }
};

#endif // GSARCH_REARRANGEMENT_H


