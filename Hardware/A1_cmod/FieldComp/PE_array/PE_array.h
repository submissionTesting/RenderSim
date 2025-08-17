#ifndef GAURAST_RE_H
#define GAURAST_RE_H

#include "GAURASTPackDef.h"
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>
#include "../PE/PE.h"
   

#pragma hls_design top
class PE_array : public match::Module {
        SC_HAS_PROCESS(PE_array);
public:
    Connections::In<ARRAY_IN_TYPE> PixelInput;          // pixel for each PE
    Connections::In<GAUSS_FEATURE_IN_TYPE> GaussInput;  // ready to be broadcasted
    Connections::In<bool> EnableIn;                    // control when the pixel is allowed
    Connections::In<bool> EnableOut;                   // control when the output is ready
    Connections::Out<ARRAY_OUT_TYPE> PEOutput;         // save the result from each PE

    // Arrays for connecting to individual PEs
    Connections::Combinational<PIXEL_POS_IN_TYPE> PixelInputs[NUM_PEs];
    Connections::Combinational<GAUSS_FEATURE_IN_TYPE> GaussInputs[NUM_PEs];
    Connections::Combinational<bool> EnableIns[NUM_PEs];
    Connections::Combinational<bool> EnableOuts[NUM_PEs];
    Connections::Combinational<OUT_TYPE> PEOutputs[NUM_PEs];

    // Array of PE instances (not PE_array)
    PE* pe[NUM_PEs];

    PE_array(sc_module_name name) : match::Module(name), 
            PixelInput("PixelInput"),
            GaussInput("GaussInput"),
            EnableIn("EnableIn"),
            EnableOut("EnableOut"),
            PEOutput("PEOutput")   
    {
       
        for (int i = 0; i < NUM_PEs; ++i) {
            pe[i] = new PE(sc_gen_unique_name("pe")); // Pass row and column index to PE

            pe[i]->clk(clk);
            pe[i]->rst(rst);
            pe[i]->PixelInput(PixelInputs[i]);
            pe[i]->GaussInput(GaussInputs[i]);
            pe[i]->EnableIn(EnableIns[i]);
            pe[i]->EnableOut(EnableOuts[i]);
            pe[i]->PEOutput(PEOutputs[i]);
        }

        SC_THREAD(pass);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(write_output);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void write_output() {
        PEOutput.Reset();
        #pragma hls_unroll yes
        for (int i = 0; i < NUM_PEs; ++i) {
            PEOutputs[i].ResetRead();
        }
        wait();

        while (true) {
            ARRAY_OUT_TYPE output;
            for (int i = 0; i < NUM_PEs; ++i) {
                PEOutputs[i].PopNB(output.x[i]);
            }   
            PEOutput.Push(output);
        }
    }

    void pass() {
        GaussInput.Reset();
        EnableIn.Reset();
        EnableOut.Reset();
        PixelInput.Reset();
        #pragma hls_unroll yes
        for (int i = 0; i < NUM_PEs; ++i) {
            GaussInputs[i].ResetWrite();  
            EnableIns[i].ResetWrite();
            EnableOuts[i].ResetWrite();
            PixelInputs[i].ResetWrite();
        }
   
        wait(2);
        // std::cout << "in pass Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
        int a = 0;
        while (true) {
           
            GAUSS_FEATURE_IN_TYPE data;
           
            if (GaussInput.PopNB(data)) {
               
                for (int i = 0; i < NUM_PEs; ++i) {
                   
                    GaussInputs[i].Push(data);  
                    

                    // std::cout << "in broadcastttttttt Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                }
            }
            bool in;
            if (EnableIn.PopNB(in)){
                for (int i = 0; i < NUM_PEs; ++i) {
                    EnableIns[i].Push(in);  
                    // std::cout << "in broadcastttttttt Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                }
            }
            bool out;
            if (EnableOut.PopNB(out)){
                for (int i = 0; i < NUM_PEs; ++i) {
                    EnableOuts[i].Push(out);  
                    // std::cout << "in broadcastttttttt Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                }
            }
            // std::cout << "in broadcastttttttt Cycle after waiting = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            
            ARRAY_IN_TYPE pixel;
            if (PixelInput.PopNB(pixel)){
                for (int i = 0; i < NUM_PEs; ++i) {
                    PixelInputs[i].Push(pixel.x[i]);
                }
            }
            
            wait();
        }
        // std::cout << "end of BBBroadcast Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
    }

};




#endif // GAURAST_RE_H
