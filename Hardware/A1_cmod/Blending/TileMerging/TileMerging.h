#ifndef GSARCH_TILEMERGING_H
#define GSARCH_TILEMERGING_H

#include "GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class TileMerging : public match::Module {
    SC_HAS_PROCESS(TileMerging);
public:
    // Input: two tile Gaussian-id lists to be merged
    Connections::In<TM_IndexVec> TileMergingInputA;
    Connections::In<TM_IndexVec> TileMergingInputB;

    // Output: hot/cold classification plus hot-address mapping
    Connections::Out<TM_HCOut> TileMergingOutput;

    // Internal: simple hot Gaussian buffer SRAM for addresses (contents are toy payload here)
    TM_SRAM<ac_int<16,false>, TM_NUM_ENTRIES, TM_NUM_BANKS> hot_sram;
    Connections::Combinational<TM_sram_msg<ac_int<16,false>>> sram_req;
    Connections::Combinational<TM_sram_msg<ac_int<16,false>>> sram_rsp;

    TileMerging(sc_module_name name) : match::Module(name),
                                       TileMergingInputA("TileMergingInputA"),
                                       TileMergingInputB("TileMergingInputB"),
                                       TileMergingOutput("TileMergingOutput"),
                                       hot_sram("hot_sram") {
        SC_THREAD(run);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        hot_sram.clk(clk);
        hot_sram.rst(rst);
        hot_sram.MemReqIn(sram_req);
        hot_sram.MemReqOut(sram_rsp);
    }

    void run() {
        TileMergingInputA.Reset();
        TileMergingInputB.Reset();
        TileMergingOutput.Reset();
        sram_req.ResetWrite();
        sram_rsp.ResetRead();

        // Persistent state across cycles
        enum {IDLE, WRITE_HOT, ISSUE_READ, WAIT_RESP, EMIT} state = IDLE;
        TM_HCOut out; // holds current tile result
        ac_int<8,false> wr_idx = 0; // index for writing hot entries to SRAM
        ac_int<16,false> pending_read_addr = 0;

        wait();
        while (1) {
            wait();

            if (state == IDLE) {
                TM_IndexVec ia, ib;
                bool have_inputs = (TileMergingInputA.PopNB(ia) && TileMergingInputB.PopNB(ib));
                if (have_inputs) {
                    // Build per-id frequency
                    ac_int<8,false> freq[SORT_NUM];
                    #pragma hls_unroll
                    for (int i = 0; i < SORT_NUM; i++) freq[i] = 0;
                    #pragma hls_unroll
                    for (int i = 0; i < TM_LIST_LEN; i++) freq[ia.id[i]] = freq[ia.id[i]] + 1;
                    #pragma hls_unroll
                    for (int i = 0; i < TM_LIST_LEN; i++) freq[ib.id[i]] = freq[ib.id[i]] + 1;

                    // Classify and stage results in 'out'
                    out.hot_count = 0;
                    out.cold_count = 0;
                    #pragma hls_unroll
                    for (int gid = 0; gid < SORT_NUM; gid++) {
                        if (freq[gid] >= TM_HOT_THRESH) {
                            ac_int<16,false> addr = (ac_int<16,false>)out.hot_count;
                            out.hot_id[(int)out.hot_count] = (SORT_ID_TYPE)gid;
                            out.hot_addr[(int)out.hot_count] = addr;
                            out.hot_count = out.hot_count + 1;
                        } else if (freq[gid] != 0) {
                            out.cold[(int)out.cold_count] = (SORT_ID_TYPE)gid;
                            out.cold_count = out.cold_count + 1;
                        }
                    }

                    wr_idx = 0;
                    state = WRITE_HOT;
                }
            } else if (state == WRITE_HOT) {
                if (wr_idx < out.hot_count) {
                    TM_sram_msg<ac_int<16,false>> wreq;
                    wreq.RW = NVUINT1(1);
                    wreq.addr = out.hot_addr[(int)wr_idx];
                    wreq.data = (ac_int<16,false>)out.hot_id[(int)wr_idx];
                    sram_req.Push(wreq);
                    wr_idx = wr_idx + 1;
                } else {
                    state = ISSUE_READ;
                }
            } else if (state == ISSUE_READ) {
                if (out.hot_count > 0) {
                    pending_read_addr = out.hot_addr[0];
                    TM_sram_msg<ac_int<16,false>> rreq;
                    rreq.RW = NVUINT1(0);
                    rreq.addr = pending_read_addr;
                    rreq.data = 0;
                    sram_req.Push(rreq);
                    state = WAIT_RESP;
                } else {
                    state = EMIT;
                }
            } else if (state == WAIT_RESP) {
                TM_sram_msg<ac_int<16,false>> rrsp;
                bool got = sram_rsp.PopNB(rrsp);
                if (got) {
                    state = EMIT;
                }
            } else { // EMIT
                TileMergingOutput.PushNB(out);
                state = IDLE;
            }
        }
    }
};

#endif // GSARCH_TILEMERGING_H


