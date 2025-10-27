#define NVHLS_VERIFY_BLOCKS (Rearrangement)
#include "Rearrangement.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>
#include <vector>
#include <map>
#include <utility>
#include <cmath>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<RU_Req> RearrangementInput;
    Connections::Combinational<RU_Update> RearrangementOutput;

    NVHLS_DESIGN(Rearrangement) dut;

    static const int N = 10;
    std::vector<RU_Req> inputs;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   RearrangementInput("RearrangementInput"),
                   RearrangementOutput("RearrangementOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.RearrangementInput(RearrangementInput);
        dut.RearrangementOutput(RearrangementOutput);

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
        RearrangementInput.ResetWrite();
        wait(10);
        inputs.clear(); inputs.reserve(N);
        // Craft inputs: duplicate addresses to test merging; span banks 0..3
        int addrs[N] = {0,1,2,3, 5,9,13, 1,5,9};
        for (int i = 0; i < N; i++) {
            RU_Req r; r.addr = addrs[i]; r.grad = FP16_TYPE(0.1*(i+1)); r.last = (i==(N-1));
            inputs.push_back(r);
            RearrangementInput.Push(r);
        }
    }

    void collect() {
        RearrangementOutput.ResetRead();
        std::map<long long, double> out_sum; // key = (bank<<32)|addr
        bool done = false;
        while (!done) {
            wait();
            RU_Update out;
            if (RearrangementOutput.PopNB(out)) {
                long long key = ((long long)out.bank << 32) | (long long)out.addr.to_int();
                out_sum[key] += out.grad.to_double();
                if (out.last) done = true;
            }
        }
        // Build expected sums per (bank,addr)
        std::map<long long, double> exp_sum;
        for (size_t i = 0; i < inputs.size(); i++) {
            int bank = inputs[i].addr.to_int() & (RU_NUM_BANKS-1);
            long long key = ((long long)bank << 32) | (long long)inputs[i].addr.to_int();
            exp_sum[key] += inputs[i].grad.to_double();
        }
        // Compare
        bool ok = true; const double tol = 1e-6;
        if (out_sum.size() != exp_sum.size()) ok = false;
        for (auto &kv : exp_sum) {
            double a = kv.second;
            double b = out_sum.count(kv.first)? out_sum[kv.first] : 1e9;
            if (std::abs(a-b) > tol) { ok = false; break; }
        }
        if (ok) {
            std::cout << "[PASS] RU accumulation matches expected across banks." << std::endl;
        } else {
            std::cout << "[FAIL] RU accumulation mismatch." << std::endl;
            NVHLS_ASSERT_MSG(false, "Rearrangement validation failed");
        }
        sc_stop();
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


