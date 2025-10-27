#ifndef IRIS_HAMAT_H
#define IRIS_HAMAT_H

#include "IRISPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>

#pragma hls_design block
class HAMAT : public match::Module {
    SC_HAS_PROCESS(HAMAT);
public:
    
    Connections::In<BSU_IN_OUT_TYPE> BSUInput;
    Connections::Out<BSU_IN_OUT_TYPE> BSUOutput;

    HAMAT(sc_module_name name) : match::Module(name),
                               BSUInput("BSUInput"),
                               BSUOutput("BSUOutput") {
        SC_THREAD(BSU_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    /*
     * Input: n-element
     * Output: n-element
     * Perform: sort 
     */
    #pragma hls_pipeline_init_interval 1
    void BSU_CALC() {
        BSUInput.Reset();
        BSUOutput.Reset();
        wait();

        while (1) {
            wait();
            
            BSU_IN_OUT_TYPE bsu_input = BSUInput.Pop();

            // Bitonic sort: the algorithm assumes SORT_NUM is a power of 2.
            // k controls the size of the subsequences (doubling each stage)
            // j controls the distance for compare–exchange within a bitonic sequence.
            for (int k = 2; k <= SORT_NUM; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    // Compare and swap for each element in the array.
                    for (int i = 0; i < SORT_NUM; i++) {
                        int ixj = i ^ j; // bitwise XOR gives the paired index

                        // Only process each pair once
                        if (ixj > i) {
                            // Determine the direction:
                            // When (i & k) == 0, sort in ascending order.
                            // Otherwise, sort in descending order.
                            if (((i & k) == 0 && (bsu_input.x[i] > bsu_input.x[ixj])) ||
                                ((i & k) != 0 && (bsu_input.x[i] < bsu_input.x[ixj]))) {
                                // Swap the two elements
                                BSU_DATA_TYPE temp = bsu_input.x[i];
                                bsu_input.x[i] = bsu_input.x[ixj];
                                bsu_input.x[ixj] = temp;
                            }
                        }
                    }
                    // Wait for one clock cycle after each j-loop stage to pipeline the operation
                    wait();
                }
            }

            // Push the sorted data to the output
            BSUOutput.Push(bsu_input);
        }
    }

};

#endif //IRIS_HAMAT_H
