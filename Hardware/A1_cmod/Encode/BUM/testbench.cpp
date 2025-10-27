#define NVHLS_VERIFY_BLOCKS (BUM)
#include "BUM.h"
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

    Connections::Combinational<BUM_IN_TYPE> BUMInput;
    Connections::Combinational<BUM_OUT_TYPE> BUMOutput;

    NVHLS_DESIGN(BUM) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   BUMInput("BUMInput"),
                   BUMOutput("BUMOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.BUMInput(BUMInput);
        dut.BUMOutput(BUMOutput);

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
        BUMInput.ResetWrite();
        wait(10);
        // Two updates to same address (merge), then one to a new address; mark last
        BUM_IN_TYPE in;
        in.addr=1; in.grad=FP16_TYPE(0.3); in.last=false; BUMInput.Push(in);
        in.addr=1; in.grad=FP16_TYPE(0.2); in.last=false; BUMInput.Push(in);
        in.addr=2; in.grad=FP16_TYPE(0.7); in.last=true;  BUMInput.Push(in);
    }

    void collect() {
        BUMOutput.ResetRead();
        int got=0; bool saw1=false,saw2=false; bool pass=true;
        while (1) {
            wait();
            BUM_OUT_TYPE o;
            if (BUMOutput.PopNB(o)) {
                got++;
                if (o.addr==1){ pass &= (fabs(o.upd.to_double() - 0.5) < 1e-3); saw1=true; }
                if (o.addr==2){ pass &= (fabs(o.upd.to_double() - 0.7) < 1e-3); saw2=true; }
                if (got==2){ if (pass && saw1 && saw2) std::cout<<"[PASS] BUM merge+commit."<<std::endl; else { std::cout<<"[FAIL] BUM"<<std::endl; NVHLS_ASSERT_MSG(false,"BUM failed"); } sc_stop(); }
            }
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


