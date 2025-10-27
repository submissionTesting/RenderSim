#define NVHLS_VERIFY_BLOCKS (TileMerging)
#include "TileMerging.h"
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

    Connections::Combinational<TM_IndexVec> TileMergingInputA;
    Connections::Combinational<TM_IndexVec> TileMergingInputB;
    Connections::Combinational<TM_HCOut>    TileMergingOutput;

    NVHLS_DESIGN(TileMerging) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   TileMergingInputA("TileMergingInputA"),
                   TileMergingInputB("TileMergingInputB"),
                   TileMergingOutput("TileMergingOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.TileMergingInputA(TileMergingInputA);
        dut.TileMergingInputB(TileMergingInputB);
        dut.TileMergingOutput(TileMergingOutput);

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
        TileMergingInputA.ResetWrite();
        TileMergingInputB.ResetWrite();
        wait(10);
        TM_IndexVec a, b;
        for (int i = 0; i < TM_LIST_LEN; i++) {
            a.id[i] = (SORT_ID_TYPE)(i % SORT_NUM);
            b.id[i] = (SORT_ID_TYPE)((i + (i % 3 == 0 ? 0 : 1)) % SORT_NUM);
        }
        TileMergingInputA.Push(a);
        TileMergingInputB.Push(b);
    }

    void collect() {
        TileMergingOutput.ResetRead();
        while (1) {
            wait();
            TM_HCOut out = TileMergingOutput.Pop();
            std::cout << "Hot (count=" << out.hot_count.to_int() << "):";
            for (int i = 0; i < out.hot_count; i++) {
                std::cout << " (id=" << out.hot_id[i].to_int() << ", addr=" << out.hot_addr[i].to_int() << ")";
            }
            std::cout << "\nCold (count=" << out.cold_count.to_int() << "):";
            for (int i = 0; i < out.cold_count; i++) {
                std::cout << " " << out.cold[i].to_int();
            }
            std::cout << std::endl;
            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


