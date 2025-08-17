#define NVHLS_VERIFY_BLOCKS (PE_array)
const int k = 13;// k is number of rounds of Gussians(each round guassian broadcast to every PE)


#include "PE_array.h"
#include <nvhls_verify.h>
//#include <mc_scverify.h>
#include "nvhls_connections.h"
#include "ac_sysc_trace.h"
#include <random>
//#include <ac_channel.h>
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>
#include <sstream>




class Top : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;


    Connections::Combinational<ARRAY_IN_TYPE> pixel_input;
    Connections::Combinational<GAUSS_FEATURE_IN_TYPE> gauss_input;
    Connections::Combinational<bool> EnableIn;
    Connections::Combinational<bool> EnableOut;
    Connections::Combinational<ARRAY_OUT_TYPE> output;

    NVHLS_DESIGN(PE_array) pe_array;


    SC_CTOR(Top) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true), rst("rst"),
                pixel_input("pixel_input"),
                gauss_input("gauss_input"),
                EnableIn("EnableIn"),
                EnableOut("EnableOut"),
                output("output"),
                pe_array("pe_array")
    {
        sc_object_tracer<sc_clock> trace_clk(clk);
        pe_array.clk(clk);
        pe_array.rst(rst);
        pe_array.GaussInput(gauss_input);
        pe_array.EnableIn(EnableIn);
        pe_array.EnableOut(EnableOut);
        pe_array.PixelInput(pixel_input);
        pe_array.PEOutput(output);
     
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
        pixel_input.ResetWrite();
        gauss_input.ResetWrite();
        EnableIn.ResetWrite();
        EnableOut.ResetWrite();
        wait(1);
     
        int a = 0;
        while (a<k){
            std::cout << "iteration:________________________________________________________" << a+1 << std::endl;
            GAUSS_FEATURE_IN_TYPE g;
            g.x[0] = POS_DATA_TYPE(300 + 100*a);
            g.x[1] = POS_DATA_TYPE(400 + 100*a);

            g.cov[0] = COV2D_DATA_TYPE(0.01+0.01*a);
            g.cov[1] = COV2D_DATA_TYPE(0.01+0.01*a);
            g.cov[2] = COV2D_DATA_TYPE(0.01+0.01*a);
            g.cov[3] = COV2D_DATA_TYPE(0.01+0.01*a);




            g.o = OPACITY_DATA_TYPE(0.01+0.01);;  // opacity




            g.color[0] = COLOR_DATA_TYPE(1.0+0.1*a);   // red
            g.color[1] = COLOR_DATA_TYPE(0.5+0.1*a);   // green
            g.color[2] = COLOR_DATA_TYPE(0.25+0.1*a);  // blue
       
           
            ARRAY_IN_TYPE p;




            // std::cout << "p too push" << p[0].x[0] << std::endl;
            // std::cout << "p too push" << p[1].x[0] << std::endl;
            // std::cout << "after p too push = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
           
            // for (int i = 0; i < NUM_PEs; ++i) {
            //     std::cout << "end of pushaaaaaaaaaaaa = " << a << std::endl;
            //     pixel_input[i].Push(p[i]);
            //     std::cout << "end of pushbbbbbbbbbbb = " << a << std::endl;


            // }
            if (a == 0){
                for (int i = 0; i < NUM_PEs; ++i) {
                    PIXEL_POS_IN_TYPE temp;
                    temp.x[0] = POS_DATA_TYPE(300 + 60* i);  // x-coordinate
                    temp.x[1] = POS_DATA_TYPE(400 + 50* i);   // y-coordinate
                    p.x[i] = temp;
                    // std::cout << "end of pushaaaaaaaaaaaa = " << p[i].x[0] << std::endl;
                    
                    // std::cout << "end of pushbbbbbbbbbbb = " << a << std::endl;
                }
                pixel_input.Push(p);
                // std::cout << "after push true" << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                EnableIn.Push(true);
            }
            else{
                EnableIn.Push(false);
            }
            if (a == k-1){
                EnableOut.Push(true);
            }
            else{
                EnableOut.Push(false);//you adjust it to be true to let output to come out after every gussian, currently only comes out after all guassian
            }
            gauss_input.Push(g);  // broadcast same Gaussian
            a++;
            wait(5);
        }
       
       
       
    }
    void collect(){
        output.ResetRead();

        
        // std::cout << "what we want " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
       
        // std::cout << "ready to receive the output " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
        bool exit = false;
        int a = 0;
        while(a<1000){
            ARRAY_OUT_TYPE outtemp = output.Pop();
            for (int i = 0; i < NUM_PEs; ++i) {
                OUT_TYPE out = outtemp.x[i];
                
                    std::cout << "PE[" << i << "] output: " << out.x[0] << std::endl;
                    exit = true;
            }
            break;
            a++;
            wait();
        }
        // std::cout << "end of output = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
        sc_stop();
       
    }
};




int sc_main(int argc, char *argv[]) {
    Top tb("tb");
    sc_start();
    return 0;
}






