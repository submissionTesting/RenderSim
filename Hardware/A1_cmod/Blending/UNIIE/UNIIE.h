#ifndef GSPROCESSOR_UNIIE_H
#define GSPROCESSOR_UNIIE_H

#include "GSPROCESSORPackDef.h"
// #include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>
#include <ac_math/ac_shift.h>
#pragma hls_design block
class UNIIE : public match::Module {
    SC_HAS_PROCESS(UNIIE);
public:
    
    Connections::In<alpha_TYPE> alphaInput;
    Connections::Out<alpha_type> UNIIEOutput;

    UNIIE(sc_module_name name) : match::Module(name),
                               alphaInput("alphaInput"),
                               UNIIEOutput("UNIIEOutput") {
        SC_THREAD(UNIIE_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Input: Mx, My, Px, Py, Cov2D, Opacity
     * Output: \alpha * Opacity
     * Perform: sort 
     */
    #pragma hls_pipeline_init_interval 1
    void UNIIE_CALC() {
        alphaInput.Reset();
        UNIIEOutput.Reset();
        wait();

        while (1) {
            wait();
            alpha_TYPE alpha_input;
            
            if (alphaInput.PopNB(alpha_input)) {
                // 2) Sum the four alpha values
                if (!((alpha_input.x[0] == alpha_type( 0 )) && (alpha_input.x[1] == alpha_type( 0 )) && (alpha_input.x[2] == alpha_type( 0 )) && (alpha_input.x[3] == alpha_type( 0 )))) {
                    alpha_type sum = alpha_input.x[0]+ alpha_input.x[1]+ alpha_input.x[2]+ alpha_input.x[3];

                    // 3) Shift right by 2
                    alpha_type shifted_sum;
                    //shifted_sum = shifted_sum >> 2;
                    // 4) If shifted_sum == 0, skip pushing; otherwise, push the result
                    ac_math::ac_shift_right(sum, 2, shifted_sum);
                    if (shifted_sum != 0) {
                        UNIIEOutput.Push(shifted_sum);
                    }
                }
            }
        }   
    }

};

#endif //GSPROCESSOR_UNIIE_H
