#define NVHLS_VERIFY_BLOCKS (RowProcessing)
#include "RowProcessing.h"
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

    Connections::Combinational<ROW_IN_TYPE> RowInput;
    Connections::Combinational<ROW_OUT_TYPE> RowOutput;

    NVHLS_DESIGN(RowProcessing) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   RowInput("RowInput"),
                   RowOutput("RowOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.RowInput(RowInput);
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
        RowInput.ResetWrite();
        wait(10);
        // Drive a short row of 3 pixels with known values
        ROW_IN_TYPE in;
        // Pixel 0: pass threshold
        in.dxpp = FP16_TYPE(1.0); in.xpp = FP16_TYPE(2.0); in.yn2 = FP16_TYPE(1.0); in.thr = FP16_TYPE(5.0);
        in.opacity = FP16_TYPE(0.5); in.color.r=FP16_TYPE(1.0); in.color.g=FP16_TYPE(0.0); in.color.b=FP16_TYPE(0.0);
        in.pix_idx = 0; in.last = false; RowInput.Push(in);
        // Pixel 1: fail threshold
        in.dxpp = FP16_TYPE(1.0); in.xpp = FP16_TYPE(0.5); in.yn2 = FP16_TYPE(0.1); in.thr = FP16_TYPE(0.2);
        in.opacity = FP16_TYPE(0.4); in.color.r=FP16_TYPE(0.0); in.color.g=FP16_TYPE(1.0); in.color.b=FP16_TYPE(0.0);
        in.pix_idx = 1; in.last = false; RowInput.Push(in);
        // Pixel 2: pass threshold and last
        in.dxpp = FP16_TYPE(2.0); in.xpp = FP16_TYPE(1.0); in.yn2 = FP16_TYPE(0.0); in.thr = FP16_TYPE(3.0);
        in.opacity = FP16_TYPE(0.5); in.color.r=FP16_TYPE(0.0); in.color.g=FP16_TYPE(0.0); in.color.b=FP16_TYPE(1.0);
        in.pix_idx = 2; in.last = true; RowInput.Push(in);
    }

    void collect() {
        RowOutput.ResetRead();
        int got = 0; FP16_TYPE T = FP16_TYPE(1.0);
        while (1) {
            wait();
            ROW_OUT_TYPE out;
            if (RowOutput.PopNB(out)) {
                // Check expected against simple model
                bool pass = true;
                if (out.pix_idx == 0) {
                    FP16_TYPE exp_r = T * FP16_TYPE(0.5) * FP16_TYPE(1.0);
                    pass &= (fabs(out.accum.r.to_double() - exp_r.to_double()) < 1e-3);
                    T = T * (FP16_TYPE(1.0) - FP16_TYPE(0.5));
                }
                if (out.pix_idx == 1) {
                    pass &= (fabs(out.accum.r.to_double() - 0.0) < 1e-6 && fabs(out.accum.g.to_double() - 0.0) < 1e-6);
                }
                if (out.pix_idx == 2) {
                    FP16_TYPE exp_b = T * FP16_TYPE(0.5) * FP16_TYPE(1.0);
                    pass &= (fabs(out.accum.b.to_double() - exp_b.to_double()) < 1e-3);
                }
                if (!pass) { std::cout << "[FAIL] RowProcessing." << std::endl; NVHLS_ASSERT_MSG(false, "RowProcessing failed"); }
                if (out.last) { std::cout << "[PASS] RowProcessing." << std::endl; sc_stop(); }
                got++;
            }
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


