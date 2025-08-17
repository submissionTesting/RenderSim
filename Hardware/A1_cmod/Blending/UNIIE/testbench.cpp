#define NVHLS_VERIFY_BLOCKS (UNIIE)
#include "UNIIE.h"
#include <nvhls_verify.h>
//#include <mc_scverify.h>
#include "nvhls_connections.h"
#include "ac_sysc_trace.h"
#include <random>
//#include <ac_channel.h>
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<alpha_TYPE> alphaInput;
    Connections::Combinational<alpha_type> UNIIEOutput;

    NVHLS_DESIGN(UNIIE) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   alphaInput("alphaInput"),
                   UNIIEOutput("REOutput"),
                   dut("dut") {

        sc_object_tracer<sc_clock> trace_clk(clk);

        dut.clk(clk);
        dut.rst(rst);
        dut.alphaInput(alphaInput);
        dut.UNIIEOutput(UNIIEOutput);

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
        alphaInput.ResetWrite();
        wait(10);

        for (int t = 0; t < 5; t++) {
            alpha_TYPE alpha;
            cout << "Input: @ timestep: " << sc_time_stamp() << ": ";
            alpha.x[0] = alpha_type( t+100 );   
            alpha.x[1] = alpha_type( t+200 );   
            alpha.x[2] = alpha_type( t+300 );   
            alpha.x[3] = alpha_type( t+400 );   
            alphaInput.Push(alpha);
        }
    }

    void collect() {
        UNIIEOutput.ResetRead();
        while (1) {
            wait(); // 1 cc

            for (int t = 0; t < 5; t++) {
                alpha_type tmp;
                tmp = UNIIEOutput.Pop();
                // compare with sample_color in vru_test.h
                cout << "UNIIEOutput: @ timestep: " << sc_time_stamp() << ": ";
                cout << tmp << " ";
                cout << endl;
            }

            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}

//#include <>
