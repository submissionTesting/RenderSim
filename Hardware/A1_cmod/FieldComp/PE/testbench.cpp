// const int NUM_PEs = 7;
// const int NUM_PEs = 5;
const int k = 5;// k is number of rounds of Gussians(each round guassian broadcast to every PE)
#define NVHLS_VERIFY_BLOCKS (PE)
#include "PE.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include "ac_sysc_trace.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>
#include <sstream>
#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;




    Connections::Combinational<PIXEL_POS_IN_TYPE> PixelInput[NUM_PEs];
    Connections::Combinational<GAUSS_FEATURE_IN_TYPE> GaussInput[NUM_PEs];
    Connections::Combinational<bool> EnableIn[NUM_PEs];
    Connections::Combinational<bool> EnableOut[NUM_PEs];
    Connections::Combinational<OUT_TYPE> PEOutput[NUM_PEs];




    NVHLS_DESIGN(PE)* pe[NUM_PEs];




    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true), rst("rst") {
        for (int i = 0; i < NUM_PEs; ++i) {
            std::stringstream ss;
            ss << "pe_" << i;
            pe[i] = new NVHLS_DESIGN(PE)(ss.str().c_str());
            pe[i]->clk(clk);
            pe[i]->rst(rst);
            pe[i]->PixelInput(PixelInput[i]);
            pe[i]->GaussInput(GaussInput[i]);
            pe[i]->EnableIn(EnableIn[i]);
            pe[i]->EnableOut(EnableOut[i]);
            pe[i]->PEOutput(PEOutput[i]);
        }




        SC_THREAD(reset);
        sensitive << clk.posedge_event();




        SC_THREAD(run);
        sensitive << clk.posedge_event();
        async_reset_signal_is(rst, false);




        SC_THREAD(collect);
        sensitive << clk.posedge_event();
        async_reset_signal_is(rst, false);
    }




    void reset() {
        rst.write(false);
        wait(10);
        rst.write(true);
    }




    void run() {
        for (int i = 0; i < NUM_PEs; ++i) {
            PixelInput[i].ResetWrite();
            GaussInput[i].ResetWrite();
            EnableIn[i].ResetWrite();
            EnableOut[i].ResetWrite();
        }



        wait(2);  // Wait for reset
       
        int a = 0;
       
        while (a<k){  
            // std::cout << "in runnnnnnnnnnnnn Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
           
            #pragma hls_unroll
            for (int i = 0; i < NUM_PEs; ++i) {
               
                GAUSS_FEATURE_IN_TYPE gauss;


               




                gauss.x[0] = POS_DATA_TYPE(300 + 100*a);
                gauss.x[1] = POS_DATA_TYPE(350 + 100*a);




                gauss.cov[0] = COV2D_DATA_TYPE(0.01+0.01*a);
                gauss.cov[1] = COV2D_DATA_TYPE(0.02 +0.01*a);
                gauss.cov[2] = COV2D_DATA_TYPE(0.03+0.005*i*a);
                gauss.cov[3] = COV2D_DATA_TYPE(0.04+0.01*i*a);




                gauss.o = OPACITY_DATA_TYPE(0.01+0.01*a);
                gauss.color[0] = COLOR_DATA_TYPE(1.0);
                gauss.color[1] = COLOR_DATA_TYPE(0.05);
                gauss.color[2] = COLOR_DATA_TYPE(0.15+0.1*a);




                GaussInput[i].PushNB(gauss);
               
                std::cout << "So this is in iteration" << a  << " " << std::endl;
               
               
            }
            
            if (a == 0){
                for (int i = 0; i < NUM_PEs; i++){
                    PIXEL_POS_IN_TYPE pixel;
                pixel.x[0] = POS_DATA_TYPE(300 + 60* i);
                pixel.x[1] = POS_DATA_TYPE(400 + 50* i);
                
                    EnableIn[i].PushNB(true);
                    PixelInput[i].PushNB(pixel);
                }
            }
            else{
                #pragma hls_unroll
                for (int i = 0; i < NUM_PEs; i++){
                    EnableIn[i].PushNB(false);
                }
            }
            if (a == k-1){
                
                #pragma hls_unroll
                for (int i = 0; i < NUM_PEs; i++){
                    // std::cout << "Is this printingwowowo" << std::endl;
                    EnableOut[i].PushNB(true);
                    // std::cout << "Is this printing" << std::endl;
                    // std::cout << "I want this" << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                }
            }
            else{
                #pragma hls_unroll
                for (int i = 0; i < NUM_PEs; i++){
                    EnableOut[i].PushNB(false);
                }
            }
            // std::cout << "QWEDQDQWD" << a << std::endl;
            a++;
           
            // std::cout << "Cycle in run = " << a << " " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            wait(); // One cycle to let all PEs begin processing
            std::cout << "Cycle in run after  = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            std::cout << "in aaaaa Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << "    " << a <<std::endl;
            
        }
    }




    void collect() {
        
        // Reset all PEOutput FIFOs
        for (int i = 0; i < NUM_PEs; ++i) {
            PEOutput[i].ResetRead();
        }
       
        // Infinite loop - runs once per clock cycle
        int c = 0;
        wait();
        
        while(1){
            wait(1); // wait for 1 clock cycle
            std::cout << "in collect Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            // Try to read and print output from each PE
            bool check = false;
            for (int i = 0; i < NUM_PEs; ++i) {
                OUT_TYPE tmp;
                bool valid = PEOutput[i].PopNB(tmp);
                if (valid){
                    cout << "Output[" << i << "] @ " << sc_time_stamp() << ": ";
                    for (int j = 0; j < 2; ++j) {
                        cout << tmp.x[j].to_double() << " ";
                    }
                    cout << std::endl;
                    check = true;
                    
                }
               
                   
            }
            if (check){
                std::cout << "Cycle in collect = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                sc_stop();
            }            
            
        
            // Optionally stop if done — or remove this to run forever
            c++;
          
        } 
            
           
        
        
    }
   
};




int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}
