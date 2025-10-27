#define NVHLS_VERIFY_BLOCKS (FRM)
#include "FRM.h"
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

    Connections::Combinational<FRM_IN_TYPE> FRMInput;
    Connections::Combinational<FRM_OUT_TYPE> FRMOutput;

    NVHLS_DESIGN(FRM) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   FRMInput("FRMInput"),
                   FRMOutput("FRMOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.FRMInput(FRMInput);
        dut.FRMOutput(FRMOutput);

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
        FRMInput.ResetWrite();
        wait(10);
        // Drive 10 requests with addresses 0..9; banks=8 → expect per timestep up to 8 outputs
        for (int i=0;i<10;i++){ FRM_IN_TYPE in; in.f = FP16_TYPE(i); in.addr = i; FRMInput.Push(in);}        
    }

    void collect() {
        FRMOutput.ResetRead();
        int got=0; int timestep=0; int seen_in_window[8]={0}; int cnt_in_window=0;
        while (1) {
            wait();
            FRM_OUT_TYPE o;
            if (FRMOutput.PopNB(o)) {
                int bank = o.addr & 7;
                // each timestep window (one outer loop iter) should have unique banks
                if (seen_in_window[bank]) { std::cout<<"[FAIL] FRM bank collision in window"<<std::endl; NVHLS_ASSERT_MSG(false,"FRM" ); }
                seen_in_window[bank]=1; cnt_in_window++;
                got++;
                if (cnt_in_window==8){ // window complete, reset tracking
                    for(int b=0;b<8;b++) seen_in_window[b]=0; cnt_in_window=0; timestep++;
                }
                if (got==10){ std::cout<<"[PASS] FRM emitted 10 reqs with per-window unique banks."<<std::endl; sc_stop(); }
            }
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


