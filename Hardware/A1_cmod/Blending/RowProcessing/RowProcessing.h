#ifndef GBU_ROWPROCESSING_H
#define GBU_ROWPROCESSING_H

#include "GBUPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class RowProcessing : public match::Module {
    SC_HAS_PROCESS(RowProcessing);
public:
    Connections::In<ROW_IN_TYPE> RowInput;
    Connections::Out<ROW_OUT_TYPE> RowOutput;

    RowProcessing(sc_module_name name) : match::Module(name),
                                         RowInput("RowInput"),
                                         RowOutput("RowOutput") {
        SC_THREAD(RowProcessing_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void RowProcessing_CALC() {
        RowInput.Reset();
        RowOutput.Reset();
        wait();

        // Running transmittance across row (FP16 as per GBU FP16 datapath)
        FP16_TYPE T = FP16_TYPE(1.0);
        // Small row pixel buffer to accumulate per-pixel color (models adders on right)
        static const int ROW_BUF_SIZE = 2; // configurable; not SRAM
        FP16_TYPE acc_r[ROW_BUF_SIZE];
        FP16_TYPE acc_g[ROW_BUF_SIZE];
        FP16_TYPE acc_b[ROW_BUF_SIZE];
        for (int i = 0; i < ROW_BUF_SIZE; i++) { acc_r[i] = FP16_TYPE(0.0); acc_g[i] = FP16_TYPE(0.0); acc_b[i] = FP16_TYPE(0.0); }

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            ROW_IN_TYPE in;
            if (RowInput.PopNB(in)) {
                // Threshold units: dxpp*xpp + (yn2)^2 < thr
                // Multipliers: 2, Adders: 1, Compare: 1
                FP16_TYPE m0 = in.dxpp * in.xpp;   // mul #1
                FP16_TYPE m1 = in.yn2 * in.yn2;    // mul #2 (y''^2)
                FP16_TYPE sum = m0 + m1;           // add
                bool pass = (sum < in.thr);        // compare

                // Prepare default output
                ROW_OUT_TYPE out;
                int idx = (int)(in.pix_idx & (ROW_BUF_SIZE - 1));
                out.accum.r = acc_r[idx];
                out.accum.g = acc_g[idx];
                out.accum.b = acc_b[idx];
                out.pix_idx = in.pix_idx;
                out.T = T;
                out.last = in.last;

                if (pass) {
                    // Color units: Ta = T*opacity; inc = Ta * color.{r,g,b}; Tnew = T*(1-opacity)
                    // Mults: 4 (Ta + 3 channels); Adds: 3 (accumulate into pixel buffer); 1 sub for (1-opacity)
                    FP16_TYPE one_minus_a = FP16_TYPE(1.0) - in.opacity; // sub
                    FP16_TYPE Ta = T * in.opacity;        // mul
                    FP16_TYPE inc_r = Ta * in.color.r;    // mul
                    FP16_TYPE inc_g = Ta * in.color.g;    // mul
                    FP16_TYPE inc_b = Ta * in.color.b;    // mul

                    // Accumulate to row pixel buffer (adders)
                    acc_r[idx] = acc_r[idx] + inc_r;      // add
                    acc_g[idx] = acc_g[idx] + inc_g;      // add
                    acc_b[idx] = acc_b[idx] + inc_b;      // add

                    out.accum.r = acc_r[idx];
                    out.accum.g = acc_g[idx];
                    out.accum.b = acc_b[idx];

                    FP16_TYPE Tnew = T * one_minus_a;     // mul
                    out.T = Tnew;
                    T = Tnew;
                }

                RowOutput.PushNB(out);

                if (in.last) {
                    T = FP16_TYPE(1.0);
                    for (int i = 0; i < ROW_BUF_SIZE; i++) { acc_r[i] = FP16_TYPE(0.0); acc_g[i] = FP16_TYPE(0.0); acc_b[i] = FP16_TYPE(0.0); }
                }
            }
        }
    }
};

#endif // GBU_ROWPROCESSING_H


