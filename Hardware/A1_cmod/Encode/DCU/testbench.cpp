#define NVHLS_VERIFY_BLOCKS (DCU)
#include "DCU.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include "ac_sysc_trace.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<DCU_IN_TYPE> DCUInput;
    Connections::Combinational<DCU_OUT_TYPE> DCUOutput;

    NVHLS_DESIGN(DCU) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   DCUInput("DCUInput"),
                   DCUOutput("DCUOutput"),
                   dut("dut") {

        sc_object_tracer<sc_clock> trace_clk(clk);

        dut.clk(clk);
        dut.rst(rst);
        dut.DCUInput(DCUInput);
        dut.DCUOutput(DCUOutput);

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
        DCUInput.ResetWrite();
        wait(10);

        // Test case 1: Identical pixels
        DCU_IN_TYPE test1;
        test1.informative_pixel.r = FP16_TYPE(128.0);
        test1.informative_pixel.g = FP16_TYPE(128.0);
        test1.informative_pixel.b = FP16_TYPE(128.0);
        test1.candidate_pixel.r = FP16_TYPE(128.0);
        test1.candidate_pixel.g = FP16_TYPE(128.0);
        test1.candidate_pixel.b = FP16_TYPE(128.0);
        
        cout << "Test Case 1: Identical pixels" << endl;
        cout << "Informative pixel: R=" << test1.informative_pixel.r 
             << ", G=" << test1.informative_pixel.g 
             << ", B=" << test1.informative_pixel.b << endl;
        cout << "Candidate pixel: R=" << test1.candidate_pixel.r 
             << ", G=" << test1.candidate_pixel.g 
             << ", B=" << test1.candidate_pixel.b << endl;
        cout << "Expected L1 distance: 0" << endl;
        
        DCUInput.Push(test1);
        wait(10);
        
        // Test case 2: All positive differences
        DCU_IN_TYPE test2;
        test2.informative_pixel.r = FP16_TYPE(100.0);
        test2.informative_pixel.g = FP16_TYPE(100.0);
        test2.informative_pixel.b = FP16_TYPE(100.0);
        test2.candidate_pixel.r = FP16_TYPE(150.0);
        test2.candidate_pixel.g = FP16_TYPE(120.0);
        test2.candidate_pixel.b = FP16_TYPE(130.0);
        
        cout << "Test Case 2: All positive differences" << endl;
        cout << "Informative pixel: R=" << test2.informative_pixel.r 
             << ", G=" << test2.informative_pixel.g 
             << ", B=" << test2.informative_pixel.b << endl;
        cout << "Candidate pixel: R=" << test2.candidate_pixel.r 
             << ", G=" << test2.candidate_pixel.g 
             << ", B=" << test2.candidate_pixel.b << endl;
        cout << "Expected L1 distance: 100 (50+20+30)" << endl;
        
        DCUInput.Push(test2);
        wait(10);
        
        // Test case 3: All negative differences
        DCU_IN_TYPE test3;
        test3.informative_pixel.r = FP16_TYPE(200.0);
        test3.informative_pixel.g = FP16_TYPE(180.0);
        test3.informative_pixel.b = FP16_TYPE(170.0);
        test3.candidate_pixel.r = FP16_TYPE(150.0);
        test3.candidate_pixel.g = FP16_TYPE(120.0);
        test3.candidate_pixel.b = FP16_TYPE(130.0);
        
        cout << "Test Case 3: All negative differences" << endl;
        cout << "Informative pixel: R=" << test3.informative_pixel.r 
             << ", G=" << test3.informative_pixel.g 
             << ", B=" << test3.informative_pixel.b << endl;
        cout << "Candidate pixel: R=" << test3.candidate_pixel.r 
             << ", G=" << test3.candidate_pixel.g 
             << ", B=" << test3.candidate_pixel.b << endl;
        cout << "Expected L1 distance: 150 (50+60+40)" << endl;
        
        DCUInput.Push(test3);
        wait(10);
        
        // Test case 4: Mixed differences
        DCU_IN_TYPE test4;
        test4.informative_pixel.r = FP16_TYPE(100.0);
        test4.informative_pixel.g = FP16_TYPE(200.0);
        test4.informative_pixel.b = FP16_TYPE(100.0);
        test4.candidate_pixel.r = FP16_TYPE(150.0);
        test4.candidate_pixel.g = FP16_TYPE(150.0);
        test4.candidate_pixel.b = FP16_TYPE(150.0);
        
        cout << "Test Case 4: Mixed differences" << endl;
        cout << "Informative pixel: R=" << test4.informative_pixel.r 
             << ", G=" << test4.informative_pixel.g 
             << ", B=" << test4.informative_pixel.b << endl;
        cout << "Candidate pixel: R=" << test4.candidate_pixel.r 
             << ", G=" << test4.candidate_pixel.g 
             << ", B=" << test4.candidate_pixel.b << endl;
        cout << "Expected L1 distance: 150 (50+50+50)" << endl;
        
        DCUInput.Push(test4);
        wait(10);
        
        // Test case 5: Edge case with large differences
        DCU_IN_TYPE test5;
        test5.informative_pixel.r = FP16_TYPE(0.0);
        test5.informative_pixel.g = FP16_TYPE(0.0);
        test5.informative_pixel.b = FP16_TYPE(0.0);
        test5.candidate_pixel.r = FP16_TYPE(255.0);
        test5.candidate_pixel.g = FP16_TYPE(255.0);
        test5.candidate_pixel.b = FP16_TYPE(255.0);
        
        cout << "Test Case 5: Edge case with large differences" << endl;
        cout << "Informative pixel: R=" << test5.informative_pixel.r 
             << ", G=" << test5.informative_pixel.g 
             << ", B=" << test5.informative_pixel.b << endl;
        cout << "Candidate pixel: R=" << test5.candidate_pixel.r 
             << ", G=" << test5.candidate_pixel.g 
             << ", B=" << test5.candidate_pixel.b << endl;
        cout << "Expected L1 distance: 765 (255+255+255)" << endl;
        
        DCUInput.Push(test5);
        wait(10);
    }

    void collect() {
        DCUOutput.ResetRead();
        wait(20);  // Wait for reset and initialization

        for (int i = 0; i < 5; i++) {  // For each of our 5 test cases
            DCU_OUT_TYPE result = DCUOutput.Pop();
            cout << "Received L1 distance: " << result.distance << endl;
            cout << "------------------------" << endl;
            wait(10);
        }

        sc_stop();
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}