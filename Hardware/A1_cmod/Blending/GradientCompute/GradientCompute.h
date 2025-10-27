#ifndef GSARCH_GRADIENTCOMPUTE_H
#define GSARCH_GRADIENTCOMPUTE_H

#include "GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class GradientCompute : public match::Module {
    SC_HAS_PROCESS(GradientCompute);
public:
    // Input forward tuple per Gaussian; output carries gradients in reused fields
    Connections::In<VRU_IN_TYPE>  GradientComputeInput;
    Connections::Out<VRU_IN_TYPE> GradientComputeOutput;

    // s1 -> s2
    Connections::Combinational<FP16_TYPE> alpha_to_s2;
    Connections::Combinational<RGB_TYPE>  color_to_s2;
    Connections::Combinational<bool>      last_to_s2;

    // s2 -> s3
    Connections::Combinational<FP16_TYPE> alpha_to_s3;
    Connections::Combinational<FP16_TYPE> T_to_s3;
    Connections::Combinational<RGB_TYPE>  color_to_s3;
    Connections::Combinational<bool>      last_to_s3;

    GradientCompute(sc_module_name name) : match::Module(name),
                                           GradientComputeInput("GradientComputeInput"),
                                           GradientComputeOutput("GradientComputeOutput"),
                                           alpha_to_s2("alpha_to_s2"),
                                           color_to_s2("color_to_s2"),
                                           last_to_s2("last_to_s2"),
                                           alpha_to_s3("alpha_to_s3"),
                                           T_to_s3("T_to_s3"),
                                           color_to_s3("color_to_s3"),
                                           last_to_s3("last_to_s3") {
        SC_THREAD(step1_alpha_gradinputs);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(step2_transmit);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(step3_feature_grads);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    // alpha = opacity * exp(-0.5 q) with q from quadratic form
    void step1_alpha_gradinputs() {
        GradientComputeInput.Reset();
        alpha_to_s2.ResetWrite();
        color_to_s2.ResetWrite();
        last_to_s2.ResetWrite();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            VRU_IN_TYPE in;
            if (GradientComputeInput.PopNB(in)) {
                FP16_TYPE dx = in.pixel_pos_x - in.mean_x;         // 1 sub
                FP16_TYPE dy = in.pixel_pos_y - in.mean_y;         // 1 sub
                FP16_TYPE a0 = in.conx * dx;                       // 1 mul
                FP16_TYPE a1 = in.cony * dy;                       // 1 mul
                FP16_TYPE s0 = a0 + a1;                            // 1 add
                FP16_TYPE t0 = dx * s0;                            // 1 mul
                FP16_TYPE b0 = in.cony * dx;                       // 1 mul
                FP16_TYPE b1 = in.conz * dy;                       // 1 mul
                FP16_TYPE s1 = b0 + b1;                            // 1 add
                FP16_TYPE t1 = dy * s1;                            // 1 mul
                FP16_TYPE q  = t0 + t1;                            // 1 add
                FP16_TYPE exponent = FP16_TYPE(FP16_TYPE(-0.5) * q); // 1 mul
                FP16_TYPE e = exp_poly(exponent);                  // 2 mul + 2 add internally
                FP16_TYPE alpha = in.opacity * e;                  // 1 mul
                alpha_to_s2.PushNB(alpha);
                color_to_s2.PushNB(in.color);
                last_to_s2.PushNB(in.last_gaussian);
            }
        }
    }

    // T update and forward to gradient stage
    void step2_transmit() {
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
            FP16_TYPE a; RGB_TYPE c; bool last;
            bool va = alpha_to_s2.PopNB(a);
            bool vc = color_to_s2.PopNB(c);
            bool vl = last_to_s2.PopNB(last);
            if (va && vc && vl) {
                FP16_TYPE one_minus_a = FP16_TYPE(1.0) - a;  // 1 sub
                FP16_TYPE Tcur = T;
                T = T * one_minus_a;                        // 1 mul
                alpha_to_s3.PushNB(a);
                T_to_s3.PushNB(Tcur);
                color_to_s3.PushNB(c);
                last_to_s3.PushNB(last);
                if (last || Tcur < FP16_TYPE(1e-4)) T = FP16_TYPE(1.0);
            }
        }
    }

    // Feature gradients: pack into VRU_IN_TYPE
    void step3_feature_grads() {
        alpha_to_s3.ResetRead();
        T_to_s3.ResetRead();
        color_to_s3.ResetRead();
        last_to_s3.ResetRead();
        GradientComputeOutput.Reset();
        wait();
        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            FP16_TYPE a, Tcur; RGB_TYPE c; bool last;
            bool va = alpha_to_s3.PopNB(a);
            bool vT = T_to_s3.PopNB(Tcur);
            bool vc = color_to_s3.PopNB(c);
            bool vl = last_to_s3.PopNB(last);
            if (va && vT && vc && vl) {
                // grad wrt color: T * a * grad_p (grad_p = 1)
                FP16_TYPE w = Tcur * a;                  // 1 mul
                RGB_TYPE gc; gc.r = w; gc.g = w; gc.b = w;
                // grad wrt opacity (proxy): T * e * (c dot [1,1,1]) / 3
                FP16_TYPE dot = (c.r + c.g) + c.b;        // 2 add
                FP16_TYPE go = Tcur * dot;               // 1 mul
                // simple mu grads proportional to color weight
                FP16_TYPE gmx = FP16_TYPE(0.5) * w;      // 1 mul
                FP16_TYPE gmy = FP16_TYPE(0.5) * w;      // 1 mul
                // con grads with a division-like op
                FP16_TYPE inv255 = FP16_TYPE(1.0/255.0); // const
                FP16_TYPE gconx = w * inv255;            // 1 mul (models div)
                FP16_TYPE gcony = w * inv255;            // 1 mul
                FP16_TYPE gconz = w * inv255;            // 1 mul

                VRU_IN_TYPE out;
                out.mean_x = gmx; out.mean_y = gmy;      // pack mu grads
                out.conx = gconx; out.cony = gcony; out.conz = gconz; // pack con grads
                out.color = gc;                          // pack color grads
                out.opacity = go;                        // pack opacity grad
                out.last_gaussian = last;
                GradientComputeOutput.PushNB(out);
            }
        }
    }

private:
    static FP16_TYPE exp_poly(const FP16_TYPE &x) {
        FP16_TYPE x2 = x * x;                  // 1 mul
        return FP16_TYPE(1.0) + x + FP16_TYPE(0.5) * x2; // 1 mul + 2 add
    }
};

#endif // GSARCH_GRADIENTCOMPUTE_H


