#define NVHLS_VERIFY_BLOCKS (DecompBinning)
#include "DecompBinning.h"
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
    Connections::Combinational<BIN_DESC_TYPE> BinDescOutput;

    NVHLS_DESIGN(DecompBinning) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   RowInput("RowInput"),
                   BinDescOutput("BinDescOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.RowInput(RowInput);
        dut.BinDescOutput(BinDescOutput);

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
        // Create a row of 6 pixels with TILE_WIDTH=16 and NUM_TILES_X=64 (from module)
        // They all map to bin 0 except we simulate two bins by setting pix_idx accordingly
        ROW_IN_TYPE in;
        in.dxpp=FP16_TYPE(1.0); in.xpp=FP16_TYPE(0.0); in.yn2=FP16_TYPE(0.0); in.thr=FP16_TYPE(1.0);
        in.opacity=FP16_TYPE(0.5); in.color.r=FP16_TYPE(1.0); in.color.g=FP16_TYPE(0.0); in.color.b=FP16_TYPE(0.0);
        for (int i=0;i<3;i++){ in.pix_idx=i; in.last=false; RowInput.Push(in);} // bin 0, run 3
        for (int i=16;i<18;i++){ in.pix_idx=i; in.last=(i==17); RowInput.Push(in);} // bin 1, run 2, last at end
    }

    void collect() {
        BinDescOutput.ResetRead();
        int got=0; bool pass=true;
        while (1) {
            wait();
            BIN_DESC_TYPE bd;
            if (BinDescOutput.PopNB(bd)) {
                if (got==0){ pass &= (bd.bin_idx==0 && bd.count==3); }
                if (got==1){ pass &= (bd.bin_idx==1 && bd.count==2); }
                got++;
                if (got==2){ if(pass) std::cout<<"[PASS] DecompBinning."<<std::endl; else {std::cout<<"[FAIL] DecompBinning."<<std::endl; NVHLS_ASSERT_MSG(false,"DecompBinning failed");} sc_stop(); }
            }
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


