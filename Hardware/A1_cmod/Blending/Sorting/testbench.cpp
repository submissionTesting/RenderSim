#define NVHLS_VERIFY_BLOCKS (Sorting)
#include "Sorting.h"
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

    Connections::Combinational<SortingVec> SortingInput;
    Connections::Combinational<SortingVec> SortingOutput;

    NVHLS_DESIGN(Sorting) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   SortingInput("SortingInput"),
                   SortingOutput("SortingOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.SortingInput(SortingInput);
        dut.SortingOutput(SortingOutput);

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
        SortingInput.ResetWrite();
        wait(10);
        for (int t = 0; t < 2; t++) {
            SortingVec in;
            for (int i = 0; i < SORT_NUM; i++) {
                int key_int = (t+1)*SORT_NUM - i; // descending to exercise ascending sort
                in.key[i] = SORT_ELEM_TYPE(key_int);
                in.id[i]  = SORT_ID_TYPE(i); // payload is original index (0..SORT_NUM-1)
            }
            SortingInput.Push(in);
        }
    }

    void collect() {
        SortingOutput.ResetRead();
        while (1) {
            wait();
            int total_errors = 0;
            for (int t = 0; t < 2; t++) {
                SortingVec out = SortingOutput.Pop();

                std::cout << "Sorted vec t=" << t << ": ";
                int expected_start = t * SORT_NUM + 1;
                for (int i = 0; i < SORT_NUM; i++) {
                    double key_d = out.key[i].to_double();
                    int key_i = static_cast<int>(key_d + 1e-6);
                    int id_i = out.id[i].to_int();
                    std::cout << "(" << key_i << ", id=" << id_i << ")" << (i + 1 == SORT_NUM ? "" : ", ");
                    int expected_key = expected_start + i;
                    int expected_id  = (SORT_NUM - 1 - i); // original index corresponding to sorted key
                    if (key_i != expected_key || id_i != expected_id) {
                        total_errors++;
                    }
                }
                std::cout << std::endl;
            }

            if (total_errors == 0) {
                std::cout << "[PASS] Sorting output matches expected ascending order with aligned payload." << std::endl;
            } else {
                std::cout << "[FAIL] Sorting mismatches: " << total_errors << std::endl;
                NVHLS_ASSERT_MSG(false, "Sorting verification failed");
            }

            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


