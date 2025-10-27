#define NVHLS_VERIFY_BLOCKS (SRAMTest)
#include "SRAMTest.h"
#include <nvhls_verify.h>
#include <systemc.h>
#include "./nlohmann_json.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

class testbench : public sc_module {
public:
    sc_clock clk; sc_signal<bool> rst;
    Connections::Combinational<ML_Input> InputData;
    Connections::Combinational<ML_Output> OutputCoeffs;
    Connections::Combinational<GBR_ConfigMsg> ConfigIn;
    Connections::Combinational<NVUINT1>       ConfigDone;
    NVHLS_DESIGN(SRAMTest) dut;
    bool dataset_loaded{false};
    int total_samples{0};

    // Dataset buffers for R^2 evaluation
    struct Rec { double idx; double dia; std::vector<double> time; std::vector<double> norm; };
    std::vector<Rec> dataset;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true), rst("rst"),
                         InputData("InputData"), OutputCoeffs("OutputCoeffs"),
                         dut("dut") {
        dut.clk(clk); dut.rst(rst);
        dut.InputData(InputData); dut.OutputCoeffs(OutputCoeffs);
        dut.ConfigIn(ConfigIn);   dut.ConfigDone(ConfigDone);
        SC_THREAD(reset); sensitive << clk.posedge_event();
        SC_THREAD(stream_config); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
        SC_THREAD(run_inference); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
        SC_THREAD(collect_results); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    }

    void reset() { rst.write(false); wait(10); rst.write(true); wait(10); std::cout << "GBR Testbench reset" << std::endl; }

    void stream_config() { /* No-op: config handled in run_inference to match Linear_Model pattern */ }

    // Helpers to evaluate polynomial and R^2
    static std::vector<double> polyval_vec(const std::vector<double>& coeffs, const std::vector<double>& x) {
        std::vector<double> y(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            double acc = 0.0;
            for (size_t k = 0; k < coeffs.size(); ++k) acc = acc * x[i] + coeffs[k];
            y[i] = acc;
        }
        return y;
    }
    static double r2_score_1d(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
        const size_t n = y_true.size(); double mean = 0.0; for (double v : y_true) mean += v; mean /= (double)n;
        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i=0;i<n;++i) { double d=y_true[i]-y_pred[i]; double e=y_true[i]-mean; ss_res += d*d; ss_tot += e*e; }
        return (ss_tot == 0.0) ? 1.0 : (1.0 - ss_res / ss_tot);
    }

    bool load_dataset_json() {
        using nlohmann::json; dataset.clear();
        const char* default_path = "./data.json";
        std::ifstream f(default_path);
        if (!f) return false;
        try {
            json js; f >> js; if (!js.is_array()) return false;
            for (const auto& rec : js) {
                Rec r; r.idx = rec.at("sensor_configs").at("electrode_index").get<double>();
                r.dia = rec.at("sensor_configs").at("electrode_diameter").get<double>();
                r.time = rec.at("df").at("time_hr").get<std::vector<double>>();
                r.norm = rec.at("df").at("normalization").get<std::vector<double>>();
                dataset.push_back(std::move(r));
            }
        } catch (...) { dataset.clear(); return false; }
        return !dataset.empty();
    }

    void run_inference() {
        InputData.ResetWrite();
        ConfigIn.ResetWrite();
        ConfigDone.ResetRead();
        wait(20);
        using nlohmann::json;
        // Load scaler (KNN-style fallback) and print scaler check
        auto parse_scale_json = [&](const std::string& path, double stdv[6], double mean[6]) -> bool {
            std::ifstream f(path);
            if (!f) return false;
            std::stringstream ss; ss << f.rdbuf(); std::string s = ss.str();
            auto scan = [&](const std::string& key, double out[6])->bool{
                size_t pos = s.find("\""+key+"\""); if (pos==std::string::npos) return false; pos = s.find('[', pos); if (pos==std::string::npos) return false; pos++;
                int k=0; std::string num; bool in=false; for (; pos<s.size() && k<6; ++pos) {
                    char c=s[pos];
                    if ((c>='0'&&c<='9')||c=='-'||c=='+'||c=='.'||c=='e'||c=='E') { num.push_back(c); in=true; }
                    else { if (in) { out[k++] = std::strtod(num.c_str(), nullptr); num.clear(); in=false; if (k==6) break; } }
                }
                return true;
            };
            return scan("std", stdv) && scan("mean", mean);
        };
        double stdv[6], mean[6];
        if (!parse_scale_json("export/y_scale.json", stdv, mean) &&
            !parse_scale_json("../Linear_Model/export/y_scale.json", stdv, mean)) {
            for (int j=0;j<6;j++){ stdv[j]=1; mean[j]=0; }
        }
        std::cout << "Scaler check: mean[0]=" << mean[0] << ", std[0]=" << stdv[0] << " (lengths: mean=6, std=6)\n";
        for (int o=0;o<6;o++) { GBR_ConfigMsg m; m.section = NVUINTW(4)(2); m.scaler_kind = NVUINTW(2)(0); m.out = NVUINTW(3)(o); m.value = ML_WEIGHT(stdv[o]); ConfigIn.Push(m); wait(1); }
        for (int o=0;o<6;o++) { GBR_ConfigMsg m; m.section = NVUINTW(4)(2); m.scaler_kind = NVUINTW(2)(1); m.out = NVUINTW(3)(o); m.value = ML_WEIGHT(mean[o]); ConfigIn.Push(m); wait(1); }

        // Load dataset JSON and prepare buffers
        nlohmann::json js;
        dataset.clear();
        {
            std::ifstream f("./data.json");
            if (f) { f >> js; if (!js.is_array()) js = nlohmann::json::array(); }
            else { js = nlohmann::json::array(); }
        }
        total_samples = (int) js.size();
        for (int i = 0; i < total_samples; ++i) {
            const auto& rj = js[i];
            const auto& sc  = rj.at("sensor_configs");
            Rec r; r.idx = sc.at("electrode_index").get<double>(); r.dia = sc.at("electrode_diameter").get<double>();
            r.time = rj.at("df").at("time_hr").get<std::vector<double>>();
            r.norm = rj.at("df").at("normalization").get<std::vector<double>>();
            dataset.push_back(std::move(r));
        }

        // Try LOOCV per-sample models
        bool use_loocv = false;
        try {
            std::ifstream fcv("./gbr_cv.json");
            if (fcv) {
                json jcv; fcv >> jcv; auto models = jcv.at("models");
                if ((int)models.size() == total_samples) {
                    use_loocv = true;
                    dataset_loaded = true;
                    for (int i=0;i<total_samples; ++i) {
                        json jm = models[i]; if (jm.is_string()) jm = json::parse(jm.get<std::string>());
                        double lr = jm.value("learning_rate", 0.1); { GBR_ConfigMsg m; m.section = NVUINTW(4)(0); m.value = ML_WEIGHT(lr); ConfigIn.Push(m); }
                        auto outs = jm.at("outputs");
                        for (size_t o=0;o<outs.size() && o<6; ++o) {
                            double init = outs[o].at("init").get<double>(); { GBR_ConfigMsg m; m.section = NVUINTW(4)(1); m.out = NVUINTW(3)(o); m.value = ML_WEIGHT(init); ConfigIn.Push(m); }
                            auto trees = outs[o].at("trees");
                            for (size_t t=0;t<trees.size() && t<64; ++t) {
                                auto ft = trees[t].at("feature").get<std::vector<int>>();
                                auto th = trees[t].at("threshold").get<std::vector<double>>();
                                auto cl = trees[t].at("children_left").get<std::vector<int>>();
                                auto cr = trees[t].at("children_right").get<std::vector<int>>();
                                auto vv = trees[t].at("value").get<std::vector<double>>();
                                size_t N = ft.size(); if (N>128) N=128;
                                for (size_t n=0;n<N; ++n) {
                                    GBR_ConfigMsg m; m.section = NVUINTW(4)(3); m.out=NVUINTW(3)(o); m.tree=NVUINTW(8)(t); m.node=NVUINTW(10)(n);
                                    int left = cl[n]; int right = cr[n]; bool leaf = (left == -1 && right == -1);
                                    m.is_leaf = NVUINT1(leaf?1:0);
                                    m.feature_idx = NVUINTW(2)((ft[n]<0)?0:ft[n]);
                                    m.left  = NVINTW(12)(left);
                                    m.right = NVINTW(12)(right);
                                    m.threshold = ML_WEIGHT(th[n]);
                                    m.leaf_value = ML_WEIGHT(vv[n]);
                                    ConfigIn.Push(m);
                                }
                            }
                        }
                        { GBR_ConfigMsg m; m.section = NVUINTW(4)(7); ConfigIn.Push(m); }
                        // Wait for ack
                        NVUINT1 done; do { wait(); } while (!ConfigDone.PopNB(done));
                        // Push corresponding input
                        cout << "Input @ " << sc_time_stamp() << endl;
                        ML_Input in; in.x[0] = ML_FLOAT(dataset[i].idx); in.x[1] = ML_FLOAT(dataset[i].dia);
                        InputData.Push(in);
                        wait(1);
                    }
                }
            }
        } catch (...) {}

        if (!use_loocv) {
            // Single-model fallback
            try {
                std::ifstream fm("./gbr.json");
                if (fm) {
                    json jm; fm >> jm; if (jm.is_string()) jm = json::parse(jm.get<std::string>());
                    double lr = jm.value("learning_rate", 0.1); { GBR_ConfigMsg m; m.section = NVUINTW(4)(0); m.value = ML_WEIGHT(lr); ConfigIn.Push(m);} 
                    auto outs = jm.at("outputs");
                    for (size_t o=0;o<outs.size() && o<6; ++o) {
                        double init = outs[o].at("init").get<double>(); { GBR_ConfigMsg m; m.section = NVUINTW(4)(1); m.out = NVUINTW(3)(o); m.value = ML_WEIGHT(init); ConfigIn.Push(m);} 
                        auto trees = outs[o].at("trees");
                        for (size_t t=0;t<trees.size() && t<64; ++t) {
                            auto ft = trees[t].at("feature").get<std::vector<int>>();
                            auto th = trees[t].at("threshold").get<std::vector<double>>();
                            auto cl = trees[t].at("children_left").get<std::vector<int>>();
                            auto cr = trees[t].at("children_right").get<std::vector<int>>();
                            auto vv = trees[t].at("value").get<std::vector<double>>();
                            size_t N = ft.size(); if (N>128) N=128;
                            for (size_t n=0;n<N; ++n) {
                                GBR_ConfigMsg m; m.section = NVUINTW(4)(3); m.out=NVUINTW(3)(o); m.tree=NVUINTW(8)(t); m.node=NVUINTW(10)(n);
                                int left = cl[n]; int right = cr[n]; bool leaf = (left == -1 && right == -1);
                                m.is_leaf = NVUINT1(leaf?1:0);
                                m.feature_idx = NVUINTW(2)((ft[n]<0)?0:ft[n]);
                                m.left  = NVINTW(12)(left);
                                m.right = NVINTW(12)(right);
                                m.threshold = ML_WEIGHT(th[n]);
                                m.leaf_value = ML_WEIGHT(vv[n]);
                                ConfigIn.Push(m);
                            }
                        }
                    }
                }
            } catch (...) {}
            { GBR_ConfigMsg m; m.section = NVUINTW(4)(7); ConfigIn.Push(m); }
            NVUINT1 done; do { wait(); } while (!ConfigDone.PopNB(done));
            // Now drive inputs and allow collector to drain
            dataset_loaded = true;
            for (int i = 0; i < total_samples; ++i) {
                ML_Input in; in.x[0] = ML_FLOAT(dataset[i].idx); in.x[1] = ML_FLOAT(dataset[i].dia);
                InputData.Push(in);
                wait(1);
            }
        }
    }

    void collect_results() {
        OutputCoeffs.ResetRead();
        // Wait until dataset is prepared (after config is done)
        while (!dataset_loaded) wait();
        std::vector<double> r2s; r2s.reserve(total_samples);
        // Collect outputs for all samples
        for (int i=0;i<total_samples; ++i) {
            wait();
            ML_Output out = OutputCoeffs.Pop();
            cout << "Output @ " << sc_time_stamp() << endl;
            std::vector<double> coeffs(6); for (int j=0;j<6;j++) coeffs[j]=out.coeff[j].to_double();
            auto y_pred = polyval_vec(coeffs, dataset[i].time);
            double r2 = r2_score_1d(dataset[i].norm, y_pred);
            r2s.push_back(r2);
        }
        double mean_r2 = 0.0; for (double v: r2s) mean_r2 += v; if (!r2s.empty()) mean_r2 /= (double) r2s.size();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Samples: " << total_samples << ", Outputs (coeff count): 6\n";
        std::cout << "Mean R² over time-series: " << mean_r2 << "\n";
        for (size_t i=0;i<std::min<size_t>(r2s.size(),6);++i) std::cout << "  R²[i=" << i << "] = " << r2s[i] << "\n";
        sc_stop();
    }
};

int sc_main(int argc, char *argv[]) { testbench tb("tb"); sc_start(); return 0; }
