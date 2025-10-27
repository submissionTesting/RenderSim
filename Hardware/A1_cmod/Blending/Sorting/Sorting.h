#ifndef GSARCH_SORTING_H
#define GSARCH_SORTING_H

#include "GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_std_float.h>

#pragma hls_design block
class Sorting : public match::Module {
    SC_HAS_PROCESS(Sorting);
public:
    // Streamed tile vector in/out, consistent with GSArch tile-scope sorting
    Connections::In<SortingVec> SortingInput;
    Connections::Out<SortingVec> SortingOutput;


    Sorting(sc_module_name name) : match::Module(name),
                                   SortingInput("SortingInput"),
                                   SortingOutput("SortingOutput") {
        SC_THREAD(sort_stage);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    // Stage-1: bitonic network (no shared state, pure transform)
    void sort_stage() {
        SortingInput.Reset();
        SortingOutput.Reset();
        wait();

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            SortingVec v;
            if (SortingInput.PopNB(v)) {
                #pragma hls_unroll
                for (int k = 2; k <= SORT_NUM; k <<= 1) {
                    #pragma hls_unroll
                    for (int j = k >> 1; j > 0; j >>= 1) {
                        #pragma hls_unroll
                        for (int i = 0; i < SORT_NUM; i++) {
                            int ixj = i ^ j;
                            if (ixj > i) {
                                bool base_asc = ((i & k) == 0);
#if (SORT_ASCENDING)
                                bool do_swap = (base_asc && (v.key[i] > v.key[ixj])) || (!base_asc && (v.key[i] < v.key[ixj]));
#else
                                bool do_swap = (base_asc && (v.key[i] < v.key[ixj])) || (!base_asc && (v.key[i] > v.key[ixj]));
#endif
                                if (do_swap) {
                                    SORT_ELEM_TYPE key_tmp = v.key[i];
                                    v.key[i] = v.key[ixj];
                                    v.key[ixj] = key_tmp;
                                    SORT_ID_TYPE id_tmp = v.id[i];
                                    v.id[i] = v.id[ixj];
                                    v.id[ixj] = id_tmp;
                                }
                            }
                        }
                    }
                }
                SortingOutput.PushNB(v);
            }
        }
    }
};

#endif // GSARCH_SORTING_H


