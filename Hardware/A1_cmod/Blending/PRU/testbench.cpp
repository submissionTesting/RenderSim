#define NVHLS_VERIFY_BLOCKS (CU)
#include "CU.h"
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

    Connections::Combinational<CU_IN_TYPE> CUInput;
    Connections::Combinational<CU_OUT_TYPE> CUOutput;

    NVHLS_DESIGN(CU) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   CUInput("CUInput"),
                   CUOutput("CUOutput"),
                   dut("dut") {

        sc_object_tracer<sc_clock> trace_clk(clk);

        dut.clk(clk);
        dut.rst(rst);
        dut.CUInput(CUInput);
        dut.CUOutput(CUOutput);

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
        CUInput.ResetWrite();
        wait(10);

        // Test case 1: Value below threshold (L1 distance < T)
        // Example of insensitive ray detection
        CU_IN_TYPE test1;
        test1.input_value = FP16_TYPE(15.0);     // L1 distance
        test1.threshold = FP16_TYPE(20.0);       // Threshold T
        test1.id = 1;                 // Ray ID
        
        cout << "Test Case 1: Value below threshold" << endl;
        cout << "Input value: " << test1.input_value << endl; 
        cout << "Threshold: " << test1.threshold << endl;
        cout << "ID: " << test1.id << endl;
        cout << "Expected result: 0 (insensitive)" << endl;
        
        CUInput.Push(test1);
        wait(10);
        
        // Test case 2: Value above threshold (L1 distance > T)
        // Example of sensitive ray detection
        CU_IN_TYPE test2;
        test2.input_value = FP16_TYPE(25.0);     // L1 distance
        test2.threshold = FP16_TYPE(20.0);       // Threshold T
        test2.id = 2;                 // Ray ID
        
        cout << "Test Case 2: Value above threshold" << endl;
        cout << "Input value: " << test2.input_value << endl; 
        cout << "Threshold: " << test2.threshold << endl;
        cout << "ID: " << test2.id << endl;
        cout << "Expected result: 1 (sensitive)" << endl;
        
        CUInput.Push(test2);
        wait(10);
        
        // Test case 3: Value exactly at threshold
        // Edge case: should be marked as sensitive
        CU_IN_TYPE test3;
        test3.input_value = FP16_TYPE(20.0);     // L1 distance
        test3.threshold = FP16_TYPE(20.0);       // Threshold T
        test3.id = 3;                 // Ray ID
        
        cout << "Test Case 3: Value exactly at threshold" << endl;
        cout << "Input value: " << test3.input_value << endl; 
        cout << "Threshold: " << test3.threshold << endl;
        cout << "ID: " << test3.id << endl;
        cout << "Expected result: 1 (sensitive)" << endl;
        
        CUInput.Push(test3);
        wait(10);
        
        // Test case 4: Zero value
        // Example of completely identical pixels
        CU_IN_TYPE test4;
        test4.input_value = FP16_TYPE(0.0);      // L1 distance
        test4.threshold = FP16_TYPE(0.1);        // Threshold T
        test4.id = 4;                 // Ray ID
        
        cout << "Test Case 4: Zero value" << endl;
        cout << "Input value: " << test4.input_value << endl; 
        cout << "Threshold: " << test4.threshold << endl;
        cout << "ID: " << test4.id << endl;
        cout << "Expected result: 0 (insensitive)" << endl;
        
        CUInput.Push(test4);
        wait(10);
        
        // Test case 5: Very large value
        // Example of highly different pixels
        CU_IN_TYPE test5;
        test5.input_value = FP16_TYPE(765.0);    // L1 distance (Max possible for RGB)
        test5.threshold = FP16_TYPE(100.0);      // Threshold T
        test5.id = 5;                 // Ray ID
        
        cout << "Test Case 5: Very large value" << endl;
        cout << "Input value: " << test5.input_value << endl; 
        cout << "Threshold: " << test5.threshold << endl;
        cout << "ID: " << test5.id << endl;
        cout << "Expected result: 1 (sensitive)" << endl;
        
        CUInput.Push(test5);
        wait(10);
        
        // Test case 6: Weight factor test
        // Example of point sensitivity testing
        CU_IN_TYPE test6;
        test6.input_value = FP16_TYPE(0.05);     // Weight factor (Ti * αi)
        test6.threshold = FP16_TYPE(0.01);       // Threshold D
        test6.id = 100;               // Point ID
        
        cout << "Test Case 6: Weight factor test" << endl;
        cout << "Input value: " << test6.input_value << endl; 
        cout << "Threshold: " << test6.threshold << endl;
        cout << "ID: " << test6.id << endl;
        cout << "Expected result: 1 (sensitive point)" << endl;
        
        CUInput.Push(test6);
        wait(10);
    }

    void collect() {
        CUOutput.ResetRead();
        wait(20);  // Wait for reset and initialization

        for (int i = 0; i < 6; i++) {  // For each of our 6 test cases
            CU_OUT_TYPE result = CUOutput.Pop();
            cout << "Received result: " << (int)result.result << endl;
            cout << "Received ID: " << result.id << endl;
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