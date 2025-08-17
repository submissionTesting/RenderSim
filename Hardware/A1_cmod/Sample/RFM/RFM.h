#ifndef IRIS_RFM_H
#define IRIS_RFM_H

#include "IRISPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>

#pragma hls_design block
class RFM : public match::Module {
    SC_HAS_PROCESS(RFM);
public:
    
    Connections::In<RFM_IN_OUT_TYPE> RFMInputX;
    Connections::In<RFM_IN_OUT_TYPE> RFMInputY;
    Connections::In<RFM_MODE_BIT> mode;
    Connections::Out<RFM_IN_OUT_TYPE> RFMOutput;

    RFM(sc_module_name name) : match::Module(name),
                               RFMInputX("RFMInputX"),
                               RFMInputY("RFMInputY"),
                               mode("mode"),
                               RFMOutput("RFMOutput") {
        SC_THREAD(RFM_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Input: x, y
     * Output: x * y
     * Perform: sort 
     */
    #pragma hls_pipeline_init_interval 1
    void RFM_CALC() {
        RFMInputX.Reset();
        RFMInputY.Reset();
        mode.Reset();
        RFMOutput.Reset();
        wait();

        while (1) {
            wait();
            
            RFM_IN_OUT_TYPE x = RFMInputX.Pop();
            RFM_IN_OUT_TYPE y = RFMInputY.Pop();
            RFM_MODE_BIT m = mode.Pop();

            
            RFMOutput.Push(bsu_input);
        }
    }

};

#endif //IRIS_RFM_H
