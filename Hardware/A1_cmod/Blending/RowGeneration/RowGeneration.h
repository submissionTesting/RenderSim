#ifndef GBU_ROWGENERATION_H
#define GBU_ROWGENERATION_H

#include "GBUPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class RowGeneration : public match::Module {
    SC_HAS_PROCESS(RowGeneration);
public:
    Connections::In<BIN_DESC_TYPE> BinDescInput;
    Connections::Out<ROW_IN_TYPE> RowOutput;

    RowGeneration(sc_module_name name) : match::Module(name),
                                         BinDescInput("BinDescInput"),
                                         RowOutput("RowOutput") {
        SC_THREAD(RowGeneration_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void RowGeneration_CALC() {
        BinDescInput.Reset();
        RowOutput.Reset();
        wait();
        // LUTs to model per-bin parameters (paper shows LUT on threshold path)
        static const int LUT_SIZE = GBU_LUT_SIZE;
        FP16_TYPE lut_dxpp   [LUT_SIZE];
        FP16_TYPE lut_xpp_base[LUT_SIZE];
        FP16_TYPE lut_yn2    [LUT_SIZE];
        FP16_TYPE lut_thr    [LUT_SIZE];
        FP16_TYPE lut_opacity[LUT_SIZE];
        RGB16     lut_color  [LUT_SIZE];

        // Initialize default LUT; optionally can be programmed via config channel (not shown)
        for (int i = 0; i < LUT_SIZE; i++) {
            lut_dxpp[i]    = FP16_TYPE(1.0);
            lut_xpp_base[i]= FP16_TYPE(0.0);
            lut_yn2[i]     = FP16_TYPE(0.0);
            lut_thr[i]     = FP16_TYPE(1e6); // default high threshold
            lut_opacity[i] = FP16_TYPE(0.5);
            lut_color[i].r = FP16_TYPE(1.0);
            lut_color[i].g = FP16_TYPE(1.0);
            lut_color[i].b = FP16_TYPE(1.0);
        }

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            BIN_DESC_TYPE bd;
            if (BinDescInput.PopNB(bd)) {
                ac_int<16,false> bin = bd.bin_idx;
                ac_int<16,false> cnt = bd.count;

                FP16_TYPE dx   = lut_dxpp   [bin & (LUT_SIZE-1)];
                FP16_TYPE x0   = lut_xpp_base[bin & (LUT_SIZE-1)];
                FP16_TYPE yn2  = lut_yn2    [bin & (LUT_SIZE-1)];
                FP16_TYPE thr  = lut_thr    [bin & (LUT_SIZE-1)];
                FP16_TYPE opa  = lut_opacity[bin & (LUT_SIZE-1)];
                RGB16     col  = lut_color  [bin & (LUT_SIZE-1)];

                ac_int<16,false> i = 0;
                bool emitting = (cnt > 0);
                while (emitting) {
                    wait();
                    ROW_IN_TYPE out;
                    out.dxpp = dx;
                    out.xpp  = x0 + FP16_TYPE(i.to_int()) * dx;
                    out.yn2  = yn2;
                    out.thr  = thr;
                    out.opacity = opa;
                    out.color   = col;
                    out.pix_idx = i;
                    out.last    = (i + 1 == cnt);
                    RowOutput.PushNB(out);

                    if (i + 1 == cnt) { emitting = false; }
                    else { i = i + 1; }
                }
            }
        }
    }
};

#endif // GBU_ROWGENERATION_H


