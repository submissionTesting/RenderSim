#define NVHLS_VERIFY_BLOCKS (GradientCompute)
#include "GradientCompute.h"
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

    Connections::Combinational<VRU_IN_TYPE> GradientComputeInput;
    Connections::Combinational<VRU_IN_TYPE> GradientComputeOutput;

    NVHLS_DESIGN(GradientCompute) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   GradientComputeInput("GradientComputeInput"),
                   GradientComputeOutput("GradientComputeOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.GradientComputeInput(GradientComputeInput);
        dut.GradientComputeOutput(GradientComputeOutput);

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
        GradientComputeInput.ResetWrite();
        wait(10);
        VRU_IN_TYPE in;
        in.pixel_pos_x = FP16_TYPE(0.0);
        in.pixel_pos_y = FP16_TYPE(0.0);
        in.mean_x      = FP16_TYPE(0.0);
        in.mean_y      = FP16_TYPE(0.0);
        in.conx        = FP16_TYPE(1.0);
        in.cony        = FP16_TYPE(0.0);
        in.conz        = FP16_TYPE(1.0);
        in.color.r     = FP16_TYPE(0.3);
        in.color.g     = FP16_TYPE(0.4);
        in.color.b     = FP16_TYPE(0.5);
        in.opacity     = FP16_TYPE(0.8);
        in.last_gaussian = true;
        GradientComputeInput.Push(in);
    }

    void collect() {
        GradientComputeOutput.ResetRead();
        while (1) {
            wait();
            VRU_IN_TYPE out = GradientComputeOutput.Pop();
            // Expected math mirrors DUT
            FP16_TYPE dx = FP16_TYPE(0.0) - FP16_TYPE(0.0);
            FP16_TYPE dy = FP16_TYPE(0.0) - FP16_TYPE(0.0);
            FP16_TYPE q  = dx*(FP16_TYPE(1.0)*dx + FP16_TYPE(0.0)*dy) + dy*(FP16_TYPE(0.0)*dx + FP16_TYPE(1.0)*dy);
            FP16_TYPE exponent = FP16_TYPE(FP16_TYPE(-0.5) * q);
            FP16_TYPE exp_poly = FP16_TYPE(1.0) + exponent + FP16_TYPE(0.5) * exponent * exponent;
            FP16_TYPE alpha = FP16_TYPE(0.8) * exp_poly;
            FP16_TYPE Tcur  = FP16_TYPE(1.0);
            FP16_TYPE w = Tcur * alpha;
            FP16_TYPE expect_color_grad = w;
            FP16_TYPE expect_go = Tcur * (FP16_TYPE(0.3) + FP16_TYPE(0.4) + FP16_TYPE(0.5));
            FP16_TYPE expect_mu = FP16_TYPE(0.5) * w;
            FP16_TYPE expect_con = w * FP16_TYPE(1.0/255.0);

            auto ad = [](double a, double b){ return std::abs(a-b); };
            bool pass = ad(out.color.r.to_double(), expect_color_grad.to_double()) < 1e-6 &&
                        ad(out.color.g.to_double(), expect_color_grad.to_double()) < 1e-6 &&
                        ad(out.color.b.to_double(), expect_color_grad.to_double()) < 1e-6 &&
                        ad(out.opacity.to_double(), expect_go.to_double()) < 1e-6 &&
                        ad(out.mean_x.to_double(), expect_mu.to_double()) < 1e-6 &&
                        ad(out.mean_y.to_double(), expect_mu.to_double()) < 1e-6 &&
                        ad(out.conx.to_double(), expect_con.to_double()) < 1e-6 &&
                        ad(out.cony.to_double(), expect_con.to_double()) < 1e-6 &&
                        ad(out.conz.to_double(), expect_con.to_double()) < 1e-6;
            std::cout << (pass?"[PASS]":"[FAIL]")
                      << " Grad: color=" << out.color.r.to_double() << "," << out.color.g.to_double() << "," << out.color.b.to_double()
                      << " go=" << out.opacity.to_double()
                      << " mu=(" << out.mean_x.to_double() << "," << out.mean_y.to_double() << ")"
                      << " con=(" << out.conx.to_double() << "," << out.cony.to_double() << "," << out.conz.to_double() << ")"
                      << std::endl;
            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


