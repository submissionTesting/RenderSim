#ifndef GBU_DECOMPBINNING_H
#define GBU_DECOMPBINNING_H

#include "GBUPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>

#pragma hls_design block
class DecompBinning : public match::Module {
    SC_HAS_PROCESS(DecompBinning);
public:
    // Input: stream of 2D Gaussians (row elements) in depth order chunks
    Connections::In<ROW_IN_TYPE> RowInput;
    // Output: bin description (tile/bin id + run length) to RowGeneration
    Connections::Out<BIN_DESC_TYPE> BinDescOutput;

    DecompBinning(sc_module_name name) : match::Module(name),
                                         RowInput("RowInput"),
                                         BinDescOutput("BinDescOutput") {
        SC_THREAD(DecompBinning_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void DecompBinning_CALC() {
        RowInput.Reset();
        BinDescOutput.Reset();
        wait();
        // Parameters (can be exposed in GBUPackDef): tile columns, pixels per tile row, bins per row
        static const int TILE_WIDTH = 16;         // pixels per tile row
        static const int NUM_TILES_X = 64;        // number of tiles per row (example)
        static const int NUM_BINS = NUM_TILES_X;  // 1 bin per tile column

        // Two-level pipeline concept: we just stream rows and bin by tile index derived from pix_idx
        // When depth chunk boundary is seen (last=true on a sentinel), we could flush. Here we rely on Row PE 'last' only for row end.

        // Current bin state
        ac_int<16,false> cur_bin = 0;
        ac_int<16,false> run_len = 0;
        bool have_bin = false;
        // One-push-per-cycle control
        bool emit_valid = false;
        BIN_DESC_TYPE emit_bd;
        bool row_flush_pending = false;

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            ROW_IN_TYPE in;
            if (RowInput.PopNB(in)) {
                // Map pixel index to tile bin
                ac_int<16,false> tile_col = (ac_int<16,false>)(in.pix_idx / TILE_WIDTH);
                if (tile_col >= NUM_BINS) tile_col = NUM_BINS - 1; // clamp

                if (!have_bin) {
                    cur_bin = tile_col; run_len = 1; have_bin = true;
                } else {
                    if (tile_col == cur_bin) {
                        run_len = run_len + 1;
                    } else {
                        // schedule emit of previous bin run (one push this cycle)
                        emit_bd.bin_idx = cur_bin; 
                        emit_bd.count = run_len; 
                        emit_valid = true;
                        // start new run with current sample
                        cur_bin = tile_col; run_len = 1;
                    }
                }

                if (in.last) {
                    // end of row: request a flush of current run after any pending emit
                    row_flush_pending = true;
                }
            }

            // Single push site at end of iteration
            bool do_push = false;
            BIN_DESC_TYPE out_bd;
            if (emit_valid) { 
                do_push = true; 
                out_bd = emit_bd; 
                emit_valid = false;
            }
            else if (row_flush_pending && have_bin) { 
                do_push = true; 
                out_bd.bin_idx = cur_bin; 
                out_bd.count = run_len; 
            }

            if (do_push) {
                BinDescOutput.PushNB(out_bd);
                if (row_flush_pending) { 
                    have_bin = false; 
                    run_len = 0; 
                    cur_bin = 0; 
                    row_flush_pending = false;
                }
            }
        }
    }
};

#endif // GBU_DECOMPBINNING_H


