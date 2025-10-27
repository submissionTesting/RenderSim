#define NVHLS_VERIFY_BLOCKS (FeatureCompute)
#include "FeatureCompute.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>
#include <iostream>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<VRU_IN_TYPE> FeatureComputeInput;
    Connections::Combinational<VRU_OUT_TYPE> FeatureComputeOutput;

    NVHLS_DESIGN(FeatureCompute) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   FeatureComputeInput("FeatureComputeInput"),
                   FeatureComputeOutput("FeatureComputeOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.FeatureComputeInput(FeatureComputeInput);
        dut.FeatureComputeOutput(FeatureComputeOutput);

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
        FeatureComputeInput.ResetWrite();
        wait(10);
        VRU_IN_TYPE in;
        // Minimal valid fields for alpha/transmittance pipeline
        in.pixel_pos_x = FP16_TYPE(0.0);
        in.pixel_pos_y = FP16_TYPE(0.0);
        in.mean_x      = FP16_TYPE(0.0);
        in.mean_y      = FP16_TYPE(0.0);
        in.conx        = FP16_TYPE(1.0);
        in.cony        = FP16_TYPE(0.0);
        in.conz        = FP16_TYPE(1.0);
        in.color.r = FP16_TYPE(0.5);
        in.color.g = FP16_TYPE(0.5);
        in.color.b = FP16_TYPE(0.5);
        in.opacity = FP16_TYPE(0.5);
        in.last_gaussian = true; // signal end so step3 emits
        FeatureComputeInput.Push(in);
    }

    void collect() {
        FeatureComputeOutput.ResetRead();
        while (1) {
            wait();
            VRU_OUT_TYPE out = FeatureComputeOutput.Pop();
            // Recreate the same input used in run() to compute expected
            FP16_TYPE pixel_pos_x = FP16_TYPE(0.0);
            FP16_TYPE pixel_pos_y = FP16_TYPE(0.0);
            FP16_TYPE mean_x      = FP16_TYPE(0.0);
            FP16_TYPE mean_y      = FP16_TYPE(0.0);
            FP16_TYPE conx        = FP16_TYPE(1.0);
            FP16_TYPE cony        = FP16_TYPE(0.0);
            FP16_TYPE conz        = FP16_TYPE(1.0);
            FP16_TYPE opacity     = FP16_TYPE(0.5);
            FP16_TYPE col         = FP16_TYPE(0.5);

            FP16_TYPE dx = pixel_pos_x - mean_x;
            FP16_TYPE dy = pixel_pos_y - mean_y;
            FP16_TYPE q  = dx * (conx * dx + cony * dy) + dy * (cony * dx + conz * dy);
            FP16_TYPE exponent = FP16_TYPE(FP16_TYPE(-0.5) * q);
            // same polynomial used in DUT: exp ≈ 1 + x + x^2/2
            FP16_TYPE exp_poly = FP16_TYPE(1.0) + exponent + FP16_TYPE(0.5) * exponent * exponent;
            FP16_TYPE alpha = opacity * exp_poly;
            FP16_TYPE expect = alpha * col; // T starts at 1, last=true

            double er = std::abs(out.color.r.to_double() - expect.to_double());
            double eg = std::abs(out.color.g.to_double() - expect.to_double());
            double eb = std::abs(out.color.b.to_double() - expect.to_double());
            bool pass = (er < 1e-6 && eg < 1e-6 && eb < 1e-6);
            std::cout << (pass ? "[PASS]" : "[FAIL]")
                      << " FC out= (" << out.color.r.to_double() << ", "
                      << out.color.g.to_double() << ", " << out.color.b.to_double()
                      << "), expect= " << expect.to_double() << std::endl;
            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


