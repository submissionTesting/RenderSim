#ifndef SRENDER_DCU_H
#define SRENDER_DCU_H

#include "SRENDERPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>

// Distance Compute Unit (DCU)
// Calculates the L1 distance between two pixels
#pragma hls_design block
class DCU : public match::Module {
    SC_HAS_PROCESS(DCU);
public:
    // Input/Output channels
    Connections::In<DCU_IN_TYPE> DCUInput;
    Connections::Out<DCU_OUT_TYPE> DCUOutput;

    // Constructor
    DCU(sc_module_name name) : match::Module(name),
                              DCUInput("DCUInput"),
                              DCUOutput("DCUOutput") {
        SC_THREAD(DCU_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Implements the Distance Compute Unit shown in Fig. 9(a)
     * Takes two pixels (informative and candidate) and calculates L1 distance
     */
    void DCU_CALC() {
        DCUInput.Reset();
        DCUOutput.Reset();
        RGB_TYPE informative_pixel;
        informative_pixel.r = FP16_TYPE(0);
        informative_pixel.g = FP16_TYPE(0);
        informative_pixel.b = FP16_TYPE(0);
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            
            // Get input pixel values
            DCU_IN_TYPE dcu_input = DCUInput.Pop();
            
            // Initialize output
            DCU_OUT_TYPE dcu_output;
            
            // Extract pixel RGB values
            informative_pixel = dcu_input.informative_pixel;
            RGB_TYPE candidate_pixel = dcu_input.candidate_pixel;
            
            // STEP 1: Compute differences for each color channel (Sub)
            FP16_TYPE diff_r = candidate_pixel.r - informative_pixel.r;
            FP16_TYPE diff_g = candidate_pixel.g - informative_pixel.g;
            FP16_TYPE diff_b = candidate_pixel.b - informative_pixel.b;
            
            // STEP 2: Absolute value circuit implementation (Abs)
            FP16_TYPE abs_r, abs_g, abs_b;
            
            // Implementation of the absolute value circuit shown in Fig. 9(a)
            // Check sign bit and conditionally negate using inverter and MUX
            bool sign_r = (diff_r < FP16_TYPE(0));
            bool sign_g = (diff_g < FP16_TYPE(0));
            bool sign_b = (diff_b < FP16_TYPE(0));
            
            // Invert if negative (inv)
            FP16_TYPE inv_r = -diff_r;
            FP16_TYPE inv_g = -diff_g;
            FP16_TYPE inv_b = -diff_b;
            
            // Select original or inverted value based on sign (MUX)
            abs_r = sign_r ? inv_r : diff_r;
            abs_g = sign_g ? inv_g : diff_g;
            abs_b = sign_b ? inv_b : diff_b;
            
            // STEP 3: Sum the absolute differences to get L1 distance (+)
            FP16_TYPE l1_distance = abs_r + abs_g + abs_b;
            
            // Set output distance
            dcu_output.distance = l1_distance;
            
            // Push output to channel
            DCUOutput.Push(dcu_output);
        }
    }
};

#endif // SRENDER_DCU_H