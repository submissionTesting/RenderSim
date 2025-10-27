#ifndef GSARCH_FEATURECOMPUTE_H
#define GSARCH_FEATURECOMPUTE_H

#include "GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class FeatureCompute : public match::Module {
    SC_HAS_PROCESS(FeatureCompute);
public:
    // I/O
    Connections::In<VRU_IN_TYPE>  FeatureComputeInput;
    Connections::Out<VRU_OUT_TYPE> FeatureComputeOutput;

    // step1 -> step2
    Connections::Combinational<FP16_TYPE> alpha_to_s2;
    Connections::Combinational<RGB_TYPE>  color_to_s2;
    Connections::Combinational<bool>      last_to_s2;

    // step2 -> step3
    Connections::Combinational<FP16_TYPE> alpha_to_s3;
    Connections::Combinational<FP16_TYPE> T_to_s3;
    Connections::Combinational<RGB_TYPE>  color_to_s3;
    Connections::Combinational<bool>      last_to_s3;

    FeatureCompute(sc_module_name name) : match::Module(name),
                                          FeatureComputeInput("FeatureComputeInput"),
                                          FeatureComputeOutput("FeatureComputeOutput"),
                                          alpha_to_s2("alpha_to_s2"),
                                          color_to_s2("color_to_s2"),
                                          last_to_s2("last_to_s2"),
                                          alpha_to_s3("alpha_to_s3"),
                                          T_to_s3("T_to_s3"),
                                          color_to_s3("color_to_s3"),
                                          last_to_s3("last_to_s3") {
        SC_THREAD(step1_alpha);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(step2_transmittance);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(step3_render);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    // Step1: Alpha computation (match VRU math and ops)
    void step1_alpha() {
        FeatureComputeInput.Reset();
        alpha_to_s2.ResetWrite();
        color_to_s2.ResetWrite();
        last_to_s2.ResetWrite();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            VRU_IN_TYPE in;
            if (FeatureComputeInput.PopNB(in)) {
                // diff = p - mu
                FP16_TYPE dx = in.pixel_pos_x - in.mean_x;  // 1 sub
                FP16_TYPE dy = in.pixel_pos_y - in.mean_y;  // 1 sub

                // quadratic form q = [dx dy] * Sigma_inv * [dx dy]^T
                // Sigma_inv laid out as (conx, cony; cony, conz)
                FP16_TYPE t0 = conx_mul(dx, in.conx, in.cony, dy); // dx*(conx*dx + cony*dy)
                FP16_TYPE t1 = cony_mul(dy, in.cony, dx, in.conz); // dy*(cony*dx + conz*dy)
                FP16_TYPE q = t0 + t1;                              // 1 add

                // exponent = -0.5 * q
                FP16_TYPE exponent = FP16_TYPE(FP16_TYPE(-0.5) * q); // 1 mul

                // exp approx
                FP16_TYPE e; exp_poly(exponent, e);

                // alpha = opacity * e
                FP16_TYPE alpha = in.opacity * e;                    // 1 mul

                // prune alpha >= 1/255
                if (alpha >= FP16_TYPE(1.0/255.0)) {
                    alpha_to_s2.PushNB(alpha);
                    color_to_s2.PushNB(in.color);
                    last_to_s2.PushNB(in.last_gaussian);
                }
            }
        }
    }

    // Step2: Transmittance update and early termination
    void step2_transmittance() {
        alpha_to_s2.ResetRead();
        color_to_s2.ResetRead();
        last_to_s2.ResetRead();
        alpha_to_s3.ResetWrite();
        T_to_s3.ResetWrite();
        color_to_s3.ResetWrite();
        last_to_s3.ResetWrite();

        FP16_TYPE T = FP16_TYPE(1.0);
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            FP16_TYPE alpha;
            RGB_TYPE color;
            bool last;
            bool va = alpha_to_s2.PopNB(alpha);
            bool vc = color_to_s2.PopNB(color);
            bool vl = last_to_s2.PopNB(last);
            if (va && vc && vl) {
                FP16_TYPE T_cur = T;
                FP16_TYPE one_minus_alpha = FP16_TYPE(1.0) - alpha;   // 1 sub
                T = T * one_minus_alpha;                               // 1 mul

                // forward to step3
                alpha_to_s3.PushNB(alpha);
                T_to_s3.PushNB(T_cur);
                color_to_s3.PushNB(color);
                last_to_s3.PushNB(last);

                if (last || (T_cur < FP16_TYPE(1e-4))) {
                    T = FP16_TYPE(1.0);
                }
            }
        }
    }

    // Step3: Volume rendering accumulation
    void step3_render() {
        alpha_to_s3.ResetRead();
        T_to_s3.ResetRead();
        color_to_s3.ResetRead();
        last_to_s3.ResetRead();
        FeatureComputeOutput.Reset();
        RGB_TYPE C; C.r = FP16_TYPE(0.0); C.g = FP16_TYPE(0.0); C.b = FP16_TYPE(0.0);
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            FP16_TYPE alpha; FP16_TYPE Tcur; RGB_TYPE col; bool last;
            bool va = alpha_to_s3.PopNB(alpha);
            bool vT = T_to_s3.PopNB(Tcur);
            bool vc = color_to_s3.PopNB(col);
            bool vl = last_to_s3.PopNB(last);
            if (va && vT && vc && vl) {
                FP16_TYPE w = Tcur * alpha;        // 1 mul
                C.r = C.r + w * col.r;             // 2 mul + 1 add (per channel)
                C.g = C.g + w * col.g;
                C.b = C.b + w * col.b;
                if (last || (Tcur < FP16_TYPE(1e-4))) {
                    VRU_OUT_TYPE out; out.color = C;
                    FeatureComputeOutput.PushNB(out);
                    C.r = FP16_TYPE(0.0); C.g = FP16_TYPE(0.0); C.b = FP16_TYPE(0.0);
                }
            }
        }
    }

private:
    // helpers to express the exact op structure (keeps multiplier/adder counts explicit)
    static FP16_TYPE conx_mul(const FP16_TYPE &dx, const FP16_TYPE &conx, const FP16_TYPE &cony, const FP16_TYPE &dy) {
        FP16_TYPE a = conx * dx;         // 1 mul
        FP16_TYPE b = cony * dy;         // 1 mul
        FP16_TYPE s = a + b;             // 1 add
        return dx * s;                   // 1 mul
    }
    static FP16_TYPE cony_mul(const FP16_TYPE &dy, const FP16_TYPE &cony, const FP16_TYPE &dx, const FP16_TYPE &conz) {
        FP16_TYPE a = cony * dx;         // 1 mul
        FP16_TYPE b = conz * dy;         // 1 mul
        FP16_TYPE s = a + b;             // 1 add
        return dy * s;                   // 1 mul
    }
    static void exp_poly(const FP16_TYPE &x, FP16_TYPE &y) {
        // 2nd-order exp approximation e^x ≈ 1 + x + x^2/2; keeps multiplier count small
        FP16_TYPE x2 = x * x;            // 1 mul
        y = FP16_TYPE(1.0) + x + FP16_TYPE(0.5) * x2; // 1 mul + 2 adds
    }
};

#endif // GSARCH_FEATURECOMPUTE_H


