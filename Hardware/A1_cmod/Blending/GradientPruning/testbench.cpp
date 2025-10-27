#define NVHLS_VERIFY_BLOCKS (GradientPruning)
#include "GradientPruning.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>
#include <vector>
#include <algorithm>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<VRU_IN_TYPE> GradientPruningInput;
    Connections::Combinational<VRU_IN_TYPE> GradientPruningOutput;

    NVHLS_DESIGN(GradientPruning) dut;

    // Test data shared between run/collect for validation
    static const int N = 6;
    FP16_TYPE vals[N];

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   GradientPruningInput("GradientPruningInput"),
                   GradientPruningOutput("GradientPruningOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.GradientPruningInput(GradientPruningInput);
        dut.GradientPruningOutput(GradientPruningOutput);

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
        GradientPruningInput.ResetWrite();
        wait(10);
        // Initialize test values (G_i carried by grad)
        vals[0] = FP16_TYPE(0.002);
        vals[1] = FP16_TYPE(0.027);
        vals[2] = FP16_TYPE(0.005);
        vals[3] = FP16_TYPE(0.001);
        vals[4] = FP16_TYPE(0.031);
        vals[5] = FP16_TYPE(0.04);
        // Push a small tile; last=true on last
        for (int i = 0; i < N; i++) {
            VRU_IN_TYPE in; in.grad = vals[i]; in.opacity = FP16_TYPE(0.0); in.mean_x=in.mean_y=in.conx=in.cony=in.conz=FP16_TYPE(0.0);
            in.color.r=in.color.g=in.color.b=FP16_TYPE(0.0);
            in.last_gaussian = (i==(N-1));
            GradientPruningInput.Push(in);
        }
    }

    void collect() {
        GradientPruningOutput.ResetRead();
        int got = 0;
        std::vector<double> outs;
        while (1) {
            wait();
            VRU_IN_TYPE out;
            if (GradientPruningOutput.PopNB(out)) {
                got++;
                outs.push_back(out.grad.to_double());
                if (out.last_gaussian) {
                    // Validate count equals min(K,N)
                    int K = GradientPruning::GP_TOPK;
                    int expect = (K < N) ? K : N;
                    bool count_ok = (got == expect);

                    // Build expected top-K multiset
                    std::vector<double> expv; expv.reserve(N);
                    for (int i = 0; i < N; i++) expv.push_back(vals[i].to_double());
                    std::sort(expv.begin(), expv.end(), [](double a, double b){ return a > b; });
                    expv.resize(expect);

                    // Check each output is in expected multiset (order-agnostic)
                    bool contents_ok = true;
                    const double tol = 1e-6;
                    for (double o : outs) {
                        bool found = false;
                        for (size_t j = 0; j < expv.size(); j++) {
                            if (std::abs(o - expv[j]) < tol) { expv.erase(expv.begin()+j); found = true; break; }
                        }
                        if (!found) { contents_ok = false; break; }
                    }

                    if (count_ok && contents_ok) {
                        std::cout << "[PASS] GradientPruning kept " << got << "/" << N << " (K=" << expect << ") correctly." << std::endl;
                    } else {
                        std::cout << "[FAIL] GradientPruning validation. got=" << got << ", expect=" << expect << ", contents_ok=" << (contents_ok?"1":"0") << std::endl;
                        NVHLS_ASSERT_MSG(false, "GradientPruning validation failed");
                    }
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


