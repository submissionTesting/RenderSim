#define NVHLS_VERIFY_BLOCKS (RFM)
#include "RFM.h"
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

    Connections::Combinational<RFM_IN_OUT_TYPE> RFMInputX;
    Connections::Combinational<RFM_IN_OUT_TYPE> RFMInputY;
    Connections::Combinational<RFM_MODE_TYPE>   RFMmode;
    Connections::Combinational<RFM_IN_OUT_TYPE> RFMOutput;

    NVHLS_DESIGN(RFM) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   RFMInputX("RFMInputX"),
                   RFMInputY("RFMInputY"),
                   RFMmode("RFMmode"),
                   RFMOutput("RFMOutput"),
                   dut("dut") {

        sc_object_tracer<sc_clock> trace_clk(clk);

        dut.clk(clk);
        dut.rst(rst);
        dut.RFMInputX(RFMInputX);
        dut.RFMInputY(RFMInputY);
        dut.RFMmode(RFMmode);
        dut.RFMOutput(RFMOutput);

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
        RFMInputX.ResetWrite();
        RFMInputY.ResetWrite();
        RFMmode.ResetWrite();
        wait(10);

        for (int t = 0; t < 5; t++) {
            RFM_IN_OUT_TYPE x = ;
            RFM_IN_OUT_TYPE y = ;
            RFM_MODE_TYPE m = ;
            cout << "RFMInput: @ timestep: " << sc_time_stamp() << ": " << x << ", " << y << ", " << m << endl; 
            RFMInputX.Push(x);
            RFMInputY.Push(y);
            RFMmode.Push(m);
        }
    }

    void collect() {
        RFMOutput.ResetRead();
        while (1) {
            wait(); // 1 cc

            for (int t = 0; t < 5; t++) {
                RFM_IN_OUT_TYPE tmp;
                tmp = RFMOutput.Pop();
                // compare with sample_color in vru_test.h
                cout << "RFMOutput: @ timestep: " << sc_time_stamp() << ": ";
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
