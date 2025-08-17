#ifndef GSPROCESSOR_IE_H
#define GSPROCESSOR_IE_H

#include "GSPROCESSORPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math.h>
#include <ac_std_float.h>


#pragma hls_design block
class IE : public match::Module {
    SC_HAS_PROCESS(IE);
public:
    // Input/Output channels
    Connections::In<VRU_IN_TYPE> VRUInput;
    Connections::In<INT12_TYPE> alpha0;
    Connections::In<INT12_TYPE> alpha1;
    Connections::In<INT12_TYPE> alpha2;
    Connections::In<INT12_TYPE> alpha3;
    Connections::Out<VRU_OUT_INT_TYPE> VRUOutput;

    // Constructor
    IE(sc_module_name name) : match::Module(name),
                              VRUInput("VRUInput"),
                              alpha0("alpha0"),
                              alpha1("alpha1"),
                              alpha2("alpha2"),
                              alpha3("alpha3"),
                              VRUOutput("VRUOutput") {
        SC_THREAD(IE_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Input: Gaussian features (mean, covariance, color, opacity)
     * Output: RGB pixel color
     * Perform: Volume rendering based on alpha computation and blending
     */
    #pragma hls_pipeline_init_interval 1
    void IE_CALC() {
        VRUInput.Reset();   
        alpha0.Reset();
        alpha1.Reset();
        alpha2.Reset();
        alpha3.Reset();
        VRUOutput.Reset();
        wait();
        
        // Initialize per-pixel transmittance and color
        INT12_TYPE transmittance = INT12_TYPE(1.0);
        RGB_INT_TYPE accumulated_color;
        accumulated_color.r = INT8_TYPE(0.0);
        accumulated_color.g = INT8_TYPE(0.0);
        accumulated_color.b = INT8_TYPE(0.0);

        while (1) {
            wait();
            
            // Get input Gaussian features
            VRU_IN_TYPE vru_input = VRUInput.Pop();
            INT12_TYPE alpha0_input = alpha0.Pop();
            INT12_TYPE alpha1_input = alpha1.Pop();
            INT12_TYPE alpha2_input = alpha2.Pop();
            INT12_TYPE alpha3_input = alpha3.Pop();
            
            // Initialize output
            VRU_OUT_TYPE vru_output;
            
            // Extract Gaussian features
            FP16_TYPE pixel_x = vru_input.pixel_pos_x;
            FP16_TYPE pixel_y = vru_input.pixel_pos_y;
            FP16_TYPE mean_x = vru_input.mean_x;
            FP16_TYPE mean_y = vru_input.mean_y;
            
            // Get 2x2 covariance matrix
            FP16_TYPE conx = vru_input.conx;
            FP16_TYPE cony = vru_input.cony;
            FP16_TYPE conz = vru_input.conz;
            
            // Get color and opacity
            RGB_TYPE gaussian_color = vru_input.color;
            FP16_TYPE opacity = vru_input.opacity;
            
            // Bitmap for subtile skipping (optional)
            // UINT8_TYPE bitmap = vru_input.bitmap;
            // UINT8_TYPE subtile_idx = vru_input.subtile_idx;

            // // Check bitmap to see if this subtile should be processed
            // bool skip_computation = ((bitmap & (1 << subtile_idx)) == 0);
            
            // // If subtile should be skipped, skip alpha computation
            // if (!skip_computation) {
                // Stage 1: Alpha Computation & Pruning
                // Compute alpha according to equation 2: α_i = o_i * exp(-0.5 * (p' - μ')^T Σ'^(-1) (p' - μ'))
                
                // Calculate (p' - μ')
                // FP16_TYPE diff_x = pixel_x - mean_x;
                // FP16_TYPE diff_y = pixel_y - mean_y;
                
                // // Calculate (p' - μ')^T Σ'^(-1) (p' - μ')
                // FP16_TYPE exponent = FP16_TYPE(FP16_TYPE(-0.5) * (
                //     diff_x * (conx * diff_x + cony * diff_y) +
                //     diff_y * (cony * diff_x + conz * diff_y)
                // ));
                
                // // Calculate alpha: α_i = o_i * exp(-0.5 * exponent)
                // FP16_TYPE alpha;
                // // ac_math::ac_exp_cordic(exponent, alpha);
                // ac_math::ac_exp_pwl(exponent, alpha);
                
                INT12_TYPE alpha = INT12_TYPE(opacity) * ((alpha0_input + alpha1_input + alpha2_input + alpha3_input)>>2);
                
                // std::cout << "Gaussian: alpha = " << alpha.to_double() << ", transmittance = " << transmittance.to_double() << std::endl;
                // Alpha pruning: check if alpha is below threshold (1/255)
                
                    // Stage 2: Early Termination
                    // Update transmittance according to equation 4: T_i+1 = T_i * (1 - α_i) = T_i - T_i * α_i
                    FP16_TYPE new_transmittance = FP16_TYPE(transmittance * (FP16_TYPE(1.0) - alpha));
                    
                    // Stage 3: Volume Rendering
                    // Accumulate color: C += T_i * α_i * c_i
                    accumulated_color.r += transmittance * alpha * INT8_TYPE(gaussian_color.r);
                    accumulated_color.g += transmittance * alpha * INT8_TYPE(gaussian_color.g);
                    accumulated_color.b += transmittance * alpha * INT8_TYPE(gaussian_color.b);
                    
                    // std::cout << "Color accumulated: (" 
                    //         << accumulated_color.r.to_double() << ", "
                    //         << accumulated_color.g.to_double() << ", "
                    //         << accumulated_color.b.to_double() << ")" << std::endl;
                    // Update transmittance for next Gaussian
                    transmittance = new_transmittance;
                    
                    // Check early termination condition (T_i+1 < 10^-4)
                    // }
                
            // }
            if (vru_input.last_gaussian) {  // || transmittance < FP16_TYPE(0.0001)) {
                // Set the final accumulated color as output
                vru_output.color = accumulated_color;
                
                // Reset transmittance and accumulated color for next pixel
                transmittance = INT12_TYPE(1.0);
                accumulated_color.r = INT8_TYPE(0.0);
                accumulated_color.g = INT8_TYPE(0.0);
                accumulated_color.b = INT8_TYPE(0.0);
                
                // Push output to channel
                VRUOutput.PushNB(vru_output);
            }
            // Early termination or finished processing all Gaussians
            
        }
    }
};

#endif