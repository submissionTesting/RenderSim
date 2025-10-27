#define NVHLS_VERIFY_BLOCKS (RowGeneration)
#include "RowGeneration.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<BIN_DESC_TYPE> BinDescInput;
    Connections::Combinational<ROW_IN_TYPE> RowOutput;

    NVHLS_DESIGN(RowGeneration) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   BinDescInput("BinDescInput"),
                   RowOutput("RowOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.BinDescInput(BinDescInput);
        dut.RowOutput(RowOutput);

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
        BinDescInput.ResetWrite();
        wait(10);
        // Drive two bins with counts 3 and 2
        BIN_DESC_TYPE b0; b0.bin_idx = 1; b0.count = 3; BinDescInput.Push(b0);
        BIN_DESC_TYPE b1; b1.bin_idx = 2; b1.count = 2; BinDescInput.Push(b1);
    }

    void collect() {
        RowOutput.ResetRead();
        int expected_total = 5;
        int got = 0; int seq = 0; bool pass = true;
        while (1) {
            wait();
            ROW_IN_TYPE out;
            if (RowOutput.PopNB(out)) {
                // Check monotonic pix_idx within each bin starting at 0, last asserted on final
                if (seq < 3) {
                    pass &= (out.pix_idx == seq);
                    if (seq < 2) pass &= (out.last == false); else pass &= (out.last == true);
                } else {
                    int s = seq - 3;
                    pass &= (out.pix_idx == s);
                    if (s < 1) pass &= (out.last == false); else pass &= (out.last == true);
                }
                got++; seq++;
                if (got == expected_total) {
                    if (pass) std::cout << "[PASS] RowGeneration." << std::endl; else { std::cout << "[FAIL] RowGeneration." << std::endl; NVHLS_ASSERT_MSG(false, "RowGeneration failed"); }
                    sc_stop();
                }
            }
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


