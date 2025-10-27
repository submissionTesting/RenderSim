#ifndef GSARCH_GRADIENTPRUNING_H
#define GSARCH_GRADIENTPRUNING_H

#include "GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class GradientPruning : public match::Module {
    SC_HAS_PROCESS(GradientPruning);
public:
    Connections::In<VRU_IN_TYPE> GradientPruningInput;
    Connections::Out<VRU_IN_TYPE> GradientPruningOutput;

    // Parameters
    static const int GP_MAX = SORT_NUM;     // max grads collected per tile
    static const int GP_TOPK = SORT_NUM/2;  // keep top-K (configurable)

    GradientPruning(sc_module_name name) : match::Module(name),
                                           GradientPruningInput("GradientPruningInput"),
                                           GradientPruningOutput("GradientPruningOutput") {
        SC_THREAD(run);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void run() {
        GradientPruningInput.Reset();
        GradientPruningOutput.Reset();
        wait();

        // FSM-based implementation (Quick-Select + prune), one wait() per step
        enum GP_State { GP_COLLECT, GP_QS_PREP, GP_QS_ADV_I, GP_QS_ADV_J, GP_QS_SWAP_OR_DECIDE, GP_EMIT_PREP, GP_EMIT };
        GP_State state = GP_COLLECT;

        VRU_IN_TYPE buf[GP_MAX];
        FP16_TYPE   mag[GP_MAX];
        int         kept_idx[GP_MAX];

        int n = 0;               // number collected in tile
        bool seen_last = false;   // last flag seen during collect

        // Quick-select state
        int K_sel = 0;            // K actually used this tile
        int qs_l = 0, qs_r = 0;   // active range [qs_l, qs_r]
        int qs_i = 0, qs_j = 0;   // partition indices
        FP16_TYPE qs_pivot = FP16_TYPE(0.0);
        FP16_TYPE kth = FP16_TYPE(0.0);

        // Emit state
        int emit_e = 0;           // emit index in kept_idx
        int kept = 0;             // number kept after pruning

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();

            if (state == GP_COLLECT) {
                VRU_IN_TYPE in;
                if (GradientPruningInput.PopNB(in)) {
                    if (n < GP_MAX) {
                        buf[n] = in;
                        mag[n] = gradient_metric(in);
                        n = n + 1;
                        if (in.last_gaussian) { seen_last = true; }
                    } else {
                        seen_last = true; // cap
                    }
                }
                if (seen_last) {
                    state = GP_QS_PREP;
                }
            } else if (state == GP_QS_PREP) {
                K_sel = (GP_TOPK < n) ? GP_TOPK : n;
                qs_l = 0; qs_r = (n > 0) ? (n - 1) : 0;
                if (K_sel == 0) { kept = 0; emit_e = 0; state = GP_EMIT_PREP; }
                else { qs_i = qs_l; qs_j = qs_r; qs_pivot = mag[(qs_l + qs_r) >> 1]; state = GP_QS_ADV_I; }
            } else if (state == GP_QS_ADV_I) {
                if (qs_i <= qs_j && mag[qs_i] > qs_pivot) { qs_i = qs_i + 1; }
                else { state = GP_QS_ADV_J; }
            } else if (state == GP_QS_ADV_J) {
                if (qs_i <= qs_j && mag[qs_j] < qs_pivot) { qs_j = qs_j - 1; }
                else { state = GP_QS_SWAP_OR_DECIDE; }
            } else if (state == GP_QS_SWAP_OR_DECIDE) {
                if (qs_i <= qs_j) {
                    // swap positions i and j (mag and buf)
                    FP16_TYPE tm = mag[qs_i]; mag[qs_i] = mag[qs_j]; mag[qs_j] = tm;
                    VRU_IN_TYPE tg = buf[qs_i]; buf[qs_i] = buf[qs_j]; buf[qs_j] = tg;
                    qs_i = qs_i + 1; qs_j = qs_j - 1; state = GP_QS_ADV_I;
                } else {
                    // decide next range
                    int left_count = (qs_j >= qs_l) ? (qs_j - qs_l + 1) : 0;
                    if (K_sel <= left_count) {
                        qs_r = qs_j;
                    } else {
                        qs_l = qs_i;
                        K_sel = K_sel - left_count;
                    }
                    if (qs_l < qs_r) {
                        qs_i = qs_l; qs_j = qs_r; qs_pivot = mag[(qs_l + qs_r) >> 1]; state = GP_QS_ADV_I;
                    } else {
                        kth = mag[qs_l]; state = GP_EMIT_PREP;
                    }
                }
            } else if (state == GP_EMIT_PREP) {
                kept = 0;
                for (int i = 0; i < n; i++) {
                    if (mag[i] >= kth) { kept_idx[kept] = i; kept = kept + 1; }
                }
                emit_e = 0; state = GP_EMIT;
            } else if (state == GP_EMIT) {
                if (emit_e < kept) {
                    int idx = kept_idx[emit_e];
                    VRU_IN_TYPE out = buf[idx];
                    out.last_gaussian = (emit_e == (kept - 1));
                    GradientPruningOutput.PushNB(out);
                    emit_e = emit_e + 1;
                } else {
                    // reset for next tile
                    n = 0; seen_last = false; state = GP_COLLECT;
                }
            }
        }
    }

private:
    static inline FP16_TYPE absf(FP16_TYPE x) {
        return (x >= FP16_TYPE(0.0)) ? x : (FP16_TYPE(0.0) - x);
    }
    static FP16_TYPE gradient_metric(VRU_IN_TYPE g) {
        // Paper uses a scalar gradient G_i per Gaussian; use provided g.grad
        return absf(g.grad);
    }
    // select_kth_desc removed in favor of streaming top-K insertion
    static VRU_IN_TYPE zero_msg() {
        VRU_IN_TYPE m;
        m.opacity = FP16_TYPE(0.0);
        m.mean_x = FP16_TYPE(0.0); m.mean_y = FP16_TYPE(0.0);
        m.conx = FP16_TYPE(0.0); m.cony = FP16_TYPE(0.0); m.conz = FP16_TYPE(0.0);
        m.color.r = FP16_TYPE(0.0); m.color.g = FP16_TYPE(0.0); m.color.b = FP16_TYPE(0.0);
        m.last_gaussian = false;
        return m;
    }
};

#endif // GSARCH_GRADIENTPRUNING_H




