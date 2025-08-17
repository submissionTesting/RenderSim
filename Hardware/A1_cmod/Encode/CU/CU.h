#ifndef SRENDER_CU_H
#define SRENDER_CU_H

#include "SRENDERPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>

// Comparison Unit (CU)
// Compares input value against threshold
#pragma hls_design block
class CU : public match::Module {
    SC_HAS_PROCESS(CU);
public:
    // Input/Output channels
    Connections::In<CU_IN_TYPE> CUInput;
    Connections::Out<CU_OUT_TYPE> CUOutput;

    // Constructor
    CU(sc_module_name name) : match::Module(name),
                             CUInput("CUInput"),
                             CUOutput("CUOutput") {
        SC_THREAD(CU_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Implements the Comparison Unit shown in Fig. 9(b)
     * Compares an input value against a threshold
     * Used for both ray sensitivity (L1 distance ≥ T) 
     * and point sensitivity (weight factor ≥ D)
     */
    void CU_CALC() {
        CUInput.Reset();
        CUOutput.Reset();
        UINT16_TYPE counter = UINT16_TYPE(0);
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            
            // Get input values
            CU_IN_TYPE cu_input;
            if (CUInput.PopNB(cu_input)) {
                // Initialize output
                CU_OUT_TYPE cu_output;
                
                // Extract input data
                FP16_TYPE input_value = cu_input.input_value;
                FP16_TYPE threshold = cu_input.threshold;
                UINT16_TYPE id = cu_input.id;
                
                // STEP 1: Compare input value with threshold (Comp)
                // Implemented as shown in Fig. 9(b)
                bool comparison_result = (input_value >= threshold);
                
                // STEP 2: Select output based on comparison (MUX)
                // Choose between 0 and 1 as result
                cu_output.result = comparison_result ? 1 : 0;
                
                // STEP 3: Pass through the ID
                cu_output.id = id;
                cu_output.cnt = counter;
                counter++;
                // Push output to channel
                CUOutput.PushNB(cu_output);
            }
        }
    }
};

#endif // SRENDER_CU_Hß