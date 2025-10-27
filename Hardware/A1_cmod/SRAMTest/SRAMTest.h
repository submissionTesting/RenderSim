//
// Gradient Boosting Regressor Model for THOR ML Inference
// Hardware implementation with SRAM integration for model loading
//

#ifndef SRAMTEST_H
#define SRAMTEST_H

#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_std_float.h>
#include <nvhls_module.h>
#include "HybridDef.h"

// ML Model data types
typedef ML_FLOAT ML_ADDR; // placeholder if needed; real SRAM types not used here

// Decision tree structure (sklearn-style; -1 children denote leaf)
class TreeNode : public nvhls_message {
public:
    NVUINTW(2) feature_idx;   // 0 or 1; ignored if is_leaf==1
    NVINTW(12) left;          // index of left child; -1 if leaf
    NVINTW(12) right;         // index of right child; -1 if leaf
    ML_FLOAT threshold;       // used when internal node
    ML_FLOAT leaf_value;      // used when leaf
    NVUINT1 is_leaf;
    AUTO_GEN_FIELD_METHODS((feature_idx, left, right, threshold, leaf_value, is_leaf))
};

// Input/Output message types for ML inference (conform to other A1_cmod models)
class ML_Input : public nvhls_message {
public:
    ML_FLOAT x[2];
    AUTO_GEN_FIELD_METHODS((x))
};

class ML_Output : public nvhls_message {
public:
    enum { NUM_COEFFS = 6 };
    ML_FLOAT coeff[NUM_COEFFS];
    AUTO_GEN_FIELD_METHODS((coeff))
};

// Configuration stream message for GBR
class GBR_ConfigMsg : public nvhls_message {
public:
    // section: 0=learning_rate (value), 1=init (out,value), 2=scaler (kind:0=std,1=mean, out,value),
    //          3=tree_node (out,tree,node, is_leaf, feature_idx, left, right, threshold, leaf_value), 7=finalize
    NVUINTW(4) section;
    NVUINTW(3) out;        // 0..5
    NVUINTW(8) tree;       // tree index per output
    NVUINTW(10) node;      // node index within tree
    NVUINT1 is_leaf;       // for section 3
    NVUINTW(2) scaler_kind;// 0=std,1=mean (section 2)
    NVUINTW(2) feature_idx;// 0 or 1 (section 3, if !is_leaf)
    NVINTW(12) left;       // child idx or -1 (section 3)
    NVINTW(12) right;      // child idx or -1 (section 3)
    ML_WEIGHT threshold;   // section 3
    ML_WEIGHT leaf_value;  // section 3
    ML_WEIGHT value;       // generic value for sections 0,1,2
    AUTO_GEN_FIELD_METHODS((section, out, tree, node, is_leaf, scaler_kind, feature_idx, left, right, threshold, leaf_value, value))
};

class SRAMTest : public match::Module {
    SC_HAS_PROCESS(SRAMTest);
public:
    Connections::In<ML_Input>      InputData;
    Connections::Out<ML_Output>    OutputCoeffs;
    Connections::In<GBR_ConfigMsg> ConfigIn;
    Connections::Out<NVUINT1>      ConfigDone;

    // GBR Model parameters (fixed sizes for hardware)
    static const int NUM_OUTPUTS = 6;
    static const int MAX_TREES_PER_OUTPUT = 64;
    static const int MAX_NODES_PER_TREE = 16;
    static const int TOTAL_NODES = NUM_OUTPUTS * MAX_TREES_PER_OUTPUT * MAX_NODES_PER_TREE;
    static const int TOTAL_TREES = NUM_OUTPUTS * MAX_TREES_PER_OUTPUT;
    static const int TREES_NUM_BANKS = 12; // 6*64*16 / 12 = 512 depth per bank

    ML_FLOAT learning_rate;
    // External SRAM-backed storage via request/response channels
    GBR_SRAM<TreeNode, 512, TREES_NUM_BANKS> *trees_sram;
    Connections::Combinational<sram_msg<TreeNode>> trees_MemReqIn;
    Connections::Combinational<sram_msg<TreeNode>> trees_MemReqOut;
    NVUINTW(8) num_trees[NUM_OUTPUTS]; // number of trees per output (small; keep as regs)
    GBR_SRAM<NVUINTW(10), 384, 1> *nodes_sram;
    Connections::Combinational<sram_msg<NVUINTW(10)>> nodes_MemReqIn;
    Connections::Combinational<sram_msg<NVUINTW(10)>> nodes_MemReqOut;
    ML_FLOAT initial_predictions[NUM_OUTPUTS];
    ML_WEIGHT ys_std[NUM_OUTPUTS];
    ML_WEIGHT ys_mean[NUM_OUTPUTS];

    SRAMTest(sc_module_name name) : match::Module(name),
                                    InputData("InputData"),
                                    OutputCoeffs("OutputCoeffs"),
                                    ConfigIn("ConfigIn"),
                                    ConfigDone("ConfigDone"),
                                    trees_MemReqIn("trees_MemReqIn"),
                                    trees_MemReqOut("trees_MemReqOut"),
                                    nodes_MemReqIn("nodes_MemReqIn"),
                                    nodes_MemReqOut("nodes_MemReqOut") {
        trees_sram = new GBR_SRAM<TreeNode, 512, TREES_NUM_BANKS>(sc_gen_unique_name("TreesSRAM"));
        trees_sram->clk(clk);
        trees_sram->rst(rst);
        trees_sram->MemReqIn(trees_MemReqIn);
        trees_sram->MemReqOut(trees_MemReqOut);

        nodes_sram = new GBR_SRAM<NVUINTW(10), 384, 1>(sc_gen_unique_name("NodesSRAM"));
        nodes_sram->clk(clk);
        nodes_sram->rst(rst);
        nodes_sram->MemReqIn(nodes_MemReqIn);
        nodes_sram->MemReqOut(nodes_MemReqOut);
        SC_THREAD(Core_Process);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    // Address calculation helpers for SRAM
    inline int node_addr(int o, int t, int n) const {
        return (o * MAX_TREES_PER_OUTPUT + t) * MAX_NODES_PER_TREE + n;
    }
    inline int tree_count_addr(int o, int t) const {
        return (o * MAX_TREES_PER_OUTPUT) + t;
    }

    // Tree prediction: fetch nodes via external SRAM by address
    ML_FLOAT predict_tree(int o, int t, int node_count, ML_FLOAT x0, ML_FLOAT x1) {
        int node_idx = 0;
        // simple guard against malformed configs
        for (int step = 0; step < node_count && node_idx >= 0 && node_idx < node_count; ++step) {
            int gaddr = node_addr(o, t, node_idx);
            sram_msg<TreeNode> req; req.RW = NVUINT1(0); req.addr = (NVUINT16) gaddr;
            trees_MemReqIn.Push(req);
            wait();
            sram_msg<TreeNode> resp = trees_MemReqOut.Pop();
            TreeNode nd = resp.data;
            if (nd.is_leaf == NVUINT1(1)) return nd.leaf_value;
            int next = node_idx;
            if (nd.feature_idx == 0) {
                next = (x0 <= nd.threshold) ? nd.left : nd.right;
            } else {
                next = (x1 <= nd.threshold) ? nd.left : nd.right;
            }
            if (next < 0 || next >= node_count) break;
            node_idx = next;
        }
        // fallback if traversal fails
        return ML_FLOAT(0);
    }

    void Core_Process() {
        InputData.Reset();
        OutputCoeffs.Reset();
        ConfigIn.Reset();
        ConfigDone.Reset();
        // Reset internal SRAM request/response channels
        trees_MemReqIn.ResetWrite();
        trees_MemReqOut.ResetRead();
        nodes_MemReqIn.ResetWrite();
        nodes_MemReqOut.ResetRead();

        learning_rate = ML_FLOAT(0.1);
        for (int o = 0; o < NUM_OUTPUTS; ++o) {
            initial_predictions[o] = ML_FLOAT(0);
            ys_std[o] = ML_WEIGHT(1);
            ys_mean[o] = ML_WEIGHT(0);
            num_trees[o] = 0;
        }
        // Initialize SRAMs once after coming out of reset
        NVUINT1 loaded_latched = NVUINT1(0);
        ML_Input input;
        ML_Output output;

        wait();
        while (1) {
            wait();
            // Config stream
            GBR_ConfigMsg msg;
            if (ConfigIn.PopNB(msg)) {
                switch (msg.section.to_uint()) {
                    case 0: { learning_rate = msg.value; break; }
                    case 1: { int o = (int)msg.out; if (o>=0 && o<NUM_OUTPUTS) initial_predictions[o] = msg.value; break; }
                    case 2: { int kind=(int)msg.scaler_kind; int j=(int)msg.out; if (j>=0&&j<NUM_OUTPUTS){ if (kind==0) ys_std[j]=msg.value; else ys_mean[j]=msg.value; } break; }
                    case 3: {
                        int o = (int)msg.out; int t = (int)msg.tree; int n = (int)msg.node;
                        if (o>=0&&o<NUM_OUTPUTS&&t>=0&&t<MAX_TREES_PER_OUTPUT&&n>=0&&n<MAX_NODES_PER_TREE) {
                            if (t + 1 > (int)num_trees[o]) num_trees[o] = (NVUINTW(8))(t + 1);
                            // update nodes_per_tree count via external SRAM
                            int tc_addr = tree_count_addr(o, t);
                            sram_msg<NVUINTW(10)> rreq; rreq.RW = NVUINT1(0); rreq.addr = (NVUINT16) tc_addr;
                            nodes_MemReqIn.Push(rreq);
                            wait();
                            sram_msg<NVUINTW(10)> rresp = nodes_MemReqOut.Pop();
                            NVUINTW(10) cur = rresp.data;
                            NVUINTW(10) newcnt = cur;
                            if ((int)cur.to_uint() < n + 1) newcnt = (NVUINTW(10))(n + 1);
                            if (newcnt.to_uint() != cur.to_uint()) {
                                sram_msg<NVUINTW(10)> wreq; wreq.RW = NVUINT1(1); wreq.addr = (NVUINT16) tc_addr; wreq.data = newcnt;
                                nodes_MemReqIn.Push(wreq);
                            }
                            // write node into SRAM
                            TreeNode nd; nd.is_leaf = msg.is_leaf; nd.feature_idx = (int)msg.feature_idx; nd.left=(int)msg.left; nd.right=(int)msg.right; nd.threshold=msg.threshold; nd.leaf_value=msg.leaf_value;
                            int gaddr = node_addr(o, t, n);
                            sram_msg<TreeNode> nreq; nreq.RW = NVUINT1(1); nreq.addr = (NVUINT16) gaddr; nreq.data = nd;
                            trees_MemReqIn.Push(nreq);
                        }
                        break;
                    }
                    case 7: { NVUINT1 one = NVUINT1(1); ConfigDone.Push(one); loaded_latched = NVUINT1(1); break; }
                    default: break;
                }
            }

            // Inference
            if (!InputData.Empty() && loaded_latched == NVUINT1(1)) {
                input = InputData.Pop();
                ML_FLOAT x0 = input.x[0] / ML_FLOAT(10.0);
                ML_FLOAT x1 = input.x[1];
                for (int out_idx = 0; out_idx < NUM_OUTPUTS; ++out_idx) {
                    ML_FLOAT pred = initial_predictions[out_idx];
                    int nt = (int)num_trees[out_idx].to_uint();
                    for (int t = 0; t < nt; ++t) {
                        int tc_addr = tree_count_addr(out_idx, t);
                        sram_msg<NVUINTW(10)> rreq; rreq.RW = NVUINT1(0); rreq.addr = (NVUINT16) tc_addr;
                        nodes_MemReqIn.Push(rreq);
                        wait();
                        sram_msg<NVUINTW(10)> rresp = nodes_MemReqOut.Pop();
                        int nn = (int) rresp.data.to_uint();
                        pred += learning_rate * predict_tree(out_idx, t, nn, x0, x1);
                    }
                    ML_FLOAT y = pred * ys_std[out_idx] + ys_mean[out_idx];
                    output.coeff[out_idx] = y;
                }
                OutputCoeffs.Push(output);
            }
        }
    }
};

#endif // SRAMTEST_H
