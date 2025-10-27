#ifndef GAURAST_VRU_H
#define GAURAST_VRU_H

#include "GAURASTPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math.h>
#include <ac_std_float.h>

#pragma hls_design block
class VRU_v1 : public match::Module {
    SC_HAS_PROCESS(VRU_v1);
public:
    // Input/Output channels
    Connections::In<VRU_IN_TYPE> VRUInput;
    Connections::Out<VRU_OUT_TYPE> VRUOutput;

    /* ---------- Channels ---------- */
    // Stage‑0 → Stage‑1
    Connections::Combinational<FP32_TYPE> diff_x_32_to_step1, diff_y_32_to_step1;
    Connections::Combinational<VRU_IN_TYPE> vru_input_to_step1;
    Connections::Combinational<FP32_TYPE> exponent_to_step2_1;
    Connections::Combinational<VRU_IN_TYPE> vru_input_to_step2_1;
    
    // Stage‑1 → Stage‑2
    Connections::Combinational<FP32_TYPE>     exponent_to_step2;
    Connections::Combinational<VRU_IN_TYPE>   vru_input_to_step2;

    // Stage‑2 → Stage‑3
    Connections::Combinational<FP32_TYPE>     alpha_out_to_step3;
    Connections::Combinational<RGB_TYPE>      gaussian_color_to_step3;
    Connections::Combinational<bool>          last_gaussian_to_step3;

    // Stage‑3 → Stage‑4
    Connections::Combinational<FP32_TYPE>     alpha_out_to_step4;
    Connections::Combinational<FP32_TYPE>     transmittance_to_step4;
    Connections::Combinational<RGB_TYPE>      gaussian_color_to_step4;
    Connections::Combinational<bool>          last_gaussian_to_step4;

    // Constructor
    VRU_v1(sc_module_name name) : match::Module(name),
                              VRUInput("VRUInput"),
                              VRUOutput("VRUOutput"),
                              diff_x_32_to_step1("diff_x_32_to_step1"),
                              diff_y_32_to_step1("diff_y_32_to_step1"),
                              vru_input_to_step1("vru_input_to_step1"),
                              exponent_to_step2_1("exponent_to_step2_1"),
                              vru_input_to_step2_1("vru_input_to_step2_1"),
                              exponent_to_step2("exponent_to_step2"),
                              vru_input_to_step2("vru_input_to_step2"),
                              alpha_out_to_step3("alpha_out_to_step3"),
                              gaussian_color_to_step3("gaussian_color_to_step3"),
                              last_gaussian_to_step3("last_gaussian_to_step3"),
                              alpha_out_to_step4("alpha_out_to_step4"),
                              transmittance_to_step4("transmittance_to_step4"),
                              gaussian_color_to_step4("gaussian_color_to_step4"),
                              last_gaussian_to_step4("last_gaussian_to_step4") {
        
        SC_THREAD(VRU_step0);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(VRU_step1);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(VRU_step1_1);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(VRU_step2);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(VRU_step3);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);  

        SC_THREAD(VRU_step4);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    #pragma hls_pipeline_init_interval 1
     void VRU_step0() {
        VRUInput.Reset();
        diff_x_32_to_step1.ResetWrite();
        diff_y_32_to_step1.ResetWrite();
        vru_input_to_step1.ResetWrite();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            
            // Get input Gaussian features
            VRU_IN_TYPE vru_input;
            if (VRUInput.PopNB(vru_input)) { 
                // Extract Gaussian features
                FP32_TYPE pixel_x = vru_input.pixel_pos_x;
                FP32_TYPE pixel_y = vru_input.pixel_pos_y;
                FP32_TYPE mean_x = vru_input.mean_x;
                FP32_TYPE mean_y = vru_input.mean_y;
                
                // Get 2x2 covariance matrix
                FP32_TYPE conx = vru_input.conx;
                FP32_TYPE cony = vru_input.cony;
                FP32_TYPE conz = vru_input.conz;
            
                // Get color and opacity
                RGB_TYPE gaussian_color = vru_input.color;
                FP32_TYPE opacity = vru_input.opacity;
                
                FP32_TYPE diff_x = pixel_x - mean_x;
                FP32_TYPE diff_y = pixel_y - mean_y;

                FP32_TYPE diff_x_32 = conx * diff_x + cony * diff_y;
                FP32_TYPE diff_y_32 = cony * diff_x + conz * diff_y;
                
                
                diff_x_32_to_step1.PushNB(diff_x_32);
                diff_y_32_to_step1.PushNB(diff_y_32);
                vru_input_to_step1.PushNB(vru_input);
            }
        }
    }

    /*
     * Input: Gaussian features (mean, covariance, color, opacity)
     * Output: RGB pixel color
     * Perform: Volume rendering based on alpha computation and blending
     */
    #pragma hls_pipeline_init_interval 1
    void VRU_step1() {
        diff_x_32_to_step1.ResetRead();
        diff_y_32_to_step1.ResetRead();
        vru_input_to_step1.ResetRead();
        exponent_to_step2_1.ResetWrite();
        vru_input_to_step2_1.ResetWrite();
        wait();


        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            
            // Get input Gaussian features
            VRU_IN_TYPE vru_input;
            FP32_TYPE diff_x_32;    
            FP32_TYPE diff_y_32;
            bool vru_input_valid = vru_input_to_step1.PopNB(vru_input);
            bool diff_x_32_valid = diff_x_32_to_step1.PopNB(diff_x_32);
            bool diff_y_32_valid = diff_y_32_to_step1.PopNB(diff_y_32);
            if (vru_input_valid && diff_x_32_valid && diff_y_32_valid) { 
                // Extract Gaussian features
                FP32_TYPE pixel_x = vru_input.pixel_pos_x;
                FP32_TYPE pixel_y = vru_input.pixel_pos_y;
                FP32_TYPE mean_x = vru_input.mean_x;
                FP32_TYPE mean_y = vru_input.mean_y;
                
                // Get 2x2 covariance matrix
                FP32_TYPE conx = vru_input.conx;
                FP32_TYPE cony = vru_input.cony;
                FP32_TYPE conz = vru_input.conz;

                
                // Get color and opacity
                RGB_TYPE gaussian_color = vru_input.color;
                FP32_TYPE opacity = vru_input.opacity;
                
                FP32_TYPE diff_x = pixel_x - mean_x;
                FP32_TYPE diff_y = pixel_y - mean_y;
                
                FP32_TYPE exponent = FP32_TYPE(FP32_TYPE(-0.5) * (
                    diff_x * (diff_x_32) +
                    diff_y * (diff_y_32)
                ));

                exponent_to_step2_1.PushNB(exponent);
                vru_input_to_step2_1.PushNB(vru_input);
            }
        }
    }

    void VRU_step1_1() {
        exponent_to_step2_1.ResetRead();
        vru_input_to_step2_1.ResetRead();
        exponent_to_step2.ResetWrite();
        vru_input_to_step2.ResetWrite();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait(); 
            FP32_TYPE exponent;
            VRU_IN_TYPE vru_input;
            bool exponent_valid = exponent_to_step2_1.PopNB(exponent);
            bool vru_input_valid = vru_input_to_step2_1.PopNB(vru_input);
            if (exponent_valid && vru_input_valid) {
                FP32_TYPE conx = vru_input.conx;
                FP32_TYPE cony = vru_input.cony;
                FP32_TYPE conz = vru_input.conz;
                FP32_TYPE det = conx*conx - cony*cony;
                FP32_TYPE inv00 = conz;
                FP32_TYPE inv01 = -cony;
                FP32_TYPE inv10 = -cony;
                FP32_TYPE inv11 = conx;
                exponent = exponent/det;
                exponent_to_step2.PushNB(exponent);
                vru_input_to_step2.PushNB(vru_input);
            }
        }
    }

    #pragma hls_pipeline_init_interval 1
    void VRU_step2() {   
        exponent_to_step2.ResetRead();
        vru_input_to_step2.ResetRead();
        alpha_out_to_step3.ResetWrite();
        gaussian_color_to_step3.ResetWrite();
        last_gaussian_to_step3.ResetWrite();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            
            FP32_TYPE exponent;
            VRU_IN_TYPE vru_input;
            bool exponent_valid = exponent_to_step2.PopNB(exponent);
            bool vru_input_valid = vru_input_to_step2.PopNB(vru_input);
            if (exponent_valid && vru_input_valid) {
                FP32_TYPE opacity = vru_input.opacity;
                FP32_TYPE alpha;
                ac_math::ac_exp_pwl(exponent, alpha);
                alpha = opacity * alpha;
                if (alpha >= FP32_TYPE(1.0/255.0)) {
                    alpha_out_to_step3.PushNB(alpha);
                    gaussian_color_to_step3.PushNB(vru_input.color);
                    last_gaussian_to_step3.PushNB(vru_input.last_gaussian);
                }
            }
        }
    }

    #pragma hls_pipeline_init_interval 1
    void VRU_step3() {
        alpha_out_to_step3.ResetRead();
        gaussian_color_to_step3.ResetRead();
        last_gaussian_to_step3.ResetRead();
        // Stage‑3 outputs
        gaussian_color_to_step4.ResetWrite();
        transmittance_to_step4.ResetWrite();
        last_gaussian_to_step4.ResetWrite();
        alpha_out_to_step4.ResetWrite();
        
        // Initialize per-pixel transmittance and color
        FP32_TYPE transmittance = FP32_TYPE(1.0);

        wait();
        
        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();

            FP32_TYPE alpha;
            RGB_TYPE gaussian_color;
            bool last_gaussian;

            bool alpha_valid = alpha_out_to_step3.PopNB(alpha);
            bool gaussian_color_valid = gaussian_color_to_step3.PopNB(gaussian_color);
            bool last_gaussian_valid = last_gaussian_to_step3.PopNB(last_gaussian);
            if (alpha_valid && gaussian_color_valid && last_gaussian_valid) {
                FP32_TYPE temp = transmittance;
                FP32_TYPE new_transmittance = FP32_TYPE(temp * (FP32_TYPE(1.0) - alpha));
                // push to Stage‑4
                gaussian_color_to_step4.PushNB(gaussian_color);
                transmittance_to_step4.PushNB(temp);
                last_gaussian_to_step4.PushNB(last_gaussian);
                alpha_out_to_step4.PushNB(alpha);
                transmittance = new_transmittance;
                if (last_gaussian || temp < FP32_TYPE(0.0001)) {
                    transmittance = FP32_TYPE(1.0);
                }
            }
        }
    }

    #pragma hls_pipeline_init_interval 1
    void VRU_step4() {
        gaussian_color_to_step4.ResetRead();
        transmittance_to_step4.ResetRead();
        last_gaussian_to_step4.ResetRead();
        alpha_out_to_step4.ResetRead();
        VRUOutput.Reset();
        
        RGB_TYPE accumulated_color;
        accumulated_color.r = FP32_TYPE(0.0);
        accumulated_color.g = FP32_TYPE(0.0);
        accumulated_color.b = FP32_TYPE(0.0);
        
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();

            FP32_TYPE transmittance;
            RGB_TYPE gaussian_color;
            bool last_gaussian;
            FP32_TYPE alpha;

            bool gaussian_color_valid = gaussian_color_to_step4.PopNB(gaussian_color);
            bool transmittance_valid = transmittance_to_step4.PopNB(transmittance);
            bool last_gaussian_valid = last_gaussian_to_step4.PopNB(last_gaussian);
            bool alpha_valid = alpha_out_to_step4.PopNB(alpha);
            if (gaussian_color_valid && transmittance_valid && last_gaussian_valid && alpha_valid) {
                // Stage 3: Volume Rendering
                // Accumulate color: C += T_i * α_i * c_i
                FP32_TYPE temp = transmittance * alpha;
                accumulated_color.r += temp * gaussian_color.r;
                accumulated_color.g += temp * gaussian_color.g;
                accumulated_color.b += temp * gaussian_color.b;
                        
                // Early termination or finished processing all Gaussians
                if (last_gaussian || transmittance < FP32_TYPE(0.0001)) {
                     // Initialize output
                    VRU_OUT_TYPE vru_output;
                    
                    // Set the final accumulated color as output
                    vru_output.color = accumulated_color;
                    
                    // Reset transmittance and accumulated color for next pixel
                    accumulated_color.r = FP32_TYPE(0.0);
                    accumulated_color.g = FP32_TYPE(0.0);
                    accumulated_color.b = FP32_TYPE(0.0);
                    
                    // Push output to channel
                    VRUOutput.PushNB(vru_output);
                }
            }
        }
    }
};

#endif
