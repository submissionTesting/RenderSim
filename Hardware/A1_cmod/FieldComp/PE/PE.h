#ifndef PE_H
#define PE_H






#include "GAURASTPackDef.h"
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>





class PE : public match::Module {
    SC_HAS_PROCESS(PE); // REQUIRED macro to enable SC_THREAD in class-style
    public:
        
   
        Connections::In<PIXEL_POS_IN_TYPE> PixelInput;
        Connections::In<GAUSS_FEATURE_IN_TYPE> GaussInput;
        Connections::In<bool> EnableIn;
        Connections::In<bool> EnableOut;
        Connections::Out<OUT_TYPE> PEOutput;
        Connections::Combinational <COV2D_DATA_TYPE> qtostep2;
        Connections::Combinational <bool> EnableOut_to_2;
        Connections::Combinational <OPACITY_DATA_TYPE> gaussinput_o_tostep2;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_0_tostep2;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_1_tostep2;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_2_tostep2;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_0_tostep3;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_1_tostep3;
        Connections::Combinational <COLOR_DATA_TYPE> gaussinput_color_2_tostep3;
        Connections::Combinational <OUT_DATA_TYPE> tmp_to_step3;
        Connections::Combinational <bool> EnableOut_to_3;
        // Connections::Combinational <POS_DATA_TYPE> result0_to_3;

        // Connections::Combinational <POS_DATA_TYPE> result1_to_3;
        // Connections::Combinational <POS_DATA_TYPE> result2_to_3;

        
        


        // Constructor
        PE(sc_module_name name) : match::Module(name),
                                  PixelInput("PixelInput"),
                                  GaussInput("GaussInput"),
                                  EnableIn("EnableIn"),
                                  EnableOut("EnableOut"),
                                  PEOutput("PEOutput"),
                                  qtostep2("qtostep2"),
                                  EnableOut_to_2("EnableOut_to_2"),
                                  gaussinput_o_tostep2("gaussinput_o_tostep2"),
                                  gaussinput_color_0_tostep2("gaussinput_color_0_tostep2"),
                                  gaussinput_color_1_tostep2("gaussinput_color_1_tostep2"),
                                  gaussinput_color_2_tostep2("gaussinput_color_2_tostep2"),
                                  gaussinput_color_0_tostep3("gaussinput_color_0_tostep3"),
                                  gaussinput_color_1_tostep3("gaussinput_color_1_tostep3"),
                                  gaussinput_color_2_tostep3("gaussinput_color_2_tostep3"),
                                  tmp_to_step3("tmp_to_step3"),
                                  EnableOut_to_3("EnableOut_to_3") {
            SC_THREAD(PE_CALC_1);
            sensitive << clk.pos();
            async_reset_signal_is(rst, false);
            SC_THREAD(PE_CALC_2);
            sensitive << clk.pos();
            async_reset_signal_is(rst, false);
            SC_THREAD(PE_CALC_3);
            sensitive << clk.pos();
            async_reset_signal_is(rst, false);
            
        }
   
        #pragma hls_pipeline_init_interval 1
        void PE_CALC_1() {//pop is read push is write
            PixelInput.Reset();
            GaussInput.Reset();
            EnableIn.Reset();
            EnableOut.Reset();
            qtostep2.ResetWrite();
            gaussinput_o_tostep2.ResetWrite();
            gaussinput_color_0_tostep2.ResetWrite();
            gaussinput_color_1_tostep2.ResetWrite();
            gaussinput_color_2_tostep2.ResetWrite();
            EnableOut_to_2.ResetWrite();

           
            
            wait(2); // wait one clock edge after reset
           
            int a = 0;
            PIXEL_POS_IN_TYPE pixel_input;
            GAUSS_FEATURE_IN_TYPE gauss_input;
            
            
            

            #pragma hls_pipeline_init_interval 1
            // std::cout << "check check check" << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            while (true){
                wait();
                // std::cout << "when I should receive true:" << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                bool Pixel_in;
                EnableIn.PopNB(Pixel_in);
                bool check;
                EnableOut.PopNB(check);
                std::cout << "chcek:" << check << std::endl;
                // std::cout << Pixel_in << std::endl;
               
                // std::cout << "in PE_CALC Cycle = " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                OUT_TYPE out;
                if ( GaussInput.PopNB(gauss_input)) {
                    // std::cout << "pixel" << std::endl;
                    // std::cout << p.x[0] << std::endl;
                    // std::cout << "guassian" << std::endl;
                    // std::cout <<  gauss_input.x[0] << std::endl;
                }
                if (Pixel_in) {//or PixelInput.PopNB(p)
                    // std::cout << "pixel" << std::endl;
                    // std::cout << p.x[0] << std::endl;
                    PixelInput.PopNB(pixel_input);
                    // std::cout << "pixel" << std::endl;
                    // std::cout <<  pixel_input.x[0] << std::endl;
                }
               // Step 1: Coordinate difference
               std::cout << "pixel: (" << pixel_input.x[0] << ", " << pixel_input.x[1] << ") "
               << "gaussian: (" << gauss_input.x[0] << ", " << gauss_input.x[1] << ")" << std::endl;
     
               POS_DATA_TYPE dx = pixel_input.x[0] - gauss_input.x[0];
               POS_DATA_TYPE dy = pixel_input.x[1] - gauss_input.x[1];


               // Step 2: Extract covariance entries
               COV2D_DATA_TYPE s00 = gauss_input.cov[0];
               COV2D_DATA_TYPE s01 = gauss_input.cov[1];
               COV2D_DATA_TYPE s10 = gauss_input.cov[2];
               COV2D_DATA_TYPE s11 = gauss_input.cov[3];


               // Step 3: Compute determinant of Sigma
               COV2D_DATA_TYPE det = s00 * s11 - s01 * s10;


               // Step 4: Compute inverse covariance matrix
               // inv(Sigma) = (1/det) * [s11, -s01; -s10, s00]
               COV2D_DATA_TYPE inv00 =  s11;
               COV2D_DATA_TYPE inv01 = -s01;
               COV2D_DATA_TYPE inv10 = -s10;
               COV2D_DATA_TYPE inv11 =  s00;
              
               // Step 5: Compute q = Δ^T * inv(Sigma) * Δ
               
               COV2D_DATA_TYPE q = (dx * (inv00 * dx + inv01 * dy) +
                                   dy * (inv10 * dx + inv11 * dy))/det;
                
                qtostep2.PushNB(q);
                gaussinput_o_tostep2.PushNB(gauss_input.o);
                gaussinput_color_0_tostep2.PushNB(gauss_input.color[0]);
                gaussinput_color_1_tostep2.PushNB(gauss_input.color[1]);
                gaussinput_color_2_tostep2.PushNB(gauss_input.color[2]);
                EnableOut_to_2.PushNB(check);

            //    std::cout << "dx" << std::endl;
            //      std::cout << dx  << std::endl;
            //      std::cout << "dy" << std::endl;
            //      std::cout << dy  << std::endl;
            //      std::cout << "inv00" << std::endl;
            //      std::cout << inv00  << std::endl;
            //      std::cout << "inv01" << std::endl;
            //      std::cout << inv01  << std::endl;
            //      std::cout << "inv10" << std::endl;
            //      std::cout << inv10  << std::endl;
                 


               // Step 3: Gaussian probability
            //    std::cout << "Cycle in pre_calc_1 " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
              
               
               
                
            }
               
            // std::cout << "after results are in" << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
            // OUT_TYPE result;
            // result.x[0] = OUT_DATA_TYPE(123); // padding
            // PEOutput.Push(result);
        }
        #pragma hls_pipeline_init_interval 1
            
        void PE_CALC_2() {//pop is read push is write
            
            qtostep2.ResetRead();
            gaussinput_o_tostep2.ResetRead();
            gaussinput_color_0_tostep2.ResetRead();
            gaussinput_color_1_tostep2.ResetRead();
            gaussinput_color_2_tostep2.ResetRead();
            EnableOut_to_2.ResetRead();
            EnableOut_to_3.ResetWrite();
            gaussinput_color_0_tostep3.ResetWrite();
            gaussinput_color_1_tostep3.ResetWrite();
            gaussinput_color_2_tostep3.ResetWrite();
            tmp_to_step3.ResetWrite();
            wait(3);
            OUT_DATA_TYPE tmp;
             OUT_DATA_TYPE prob;
             OUT_TYPE result;
             OUT_DATA_TYPE alpha_reg = OUT_DATA_TYPE(1);
             COV2D_DATA_TYPE q;
             OPACITY_DATA_TYPE o;
             COLOR_DATA_TYPE color0;
             COLOR_DATA_TYPE color1;
             COLOR_DATA_TYPE color2;
             bool ee;
             #pragma hls_pipeline_init_interval 1
            while(1){
                wait();
                bool c = qtostep2.PopNB(q);
                bool gaussinput_o_valid = gaussinput_o_tostep2.PopNB(o);
                bool gaussinput_color_0_valid = gaussinput_color_0_tostep2.PopNB(color0);
                bool gaussinput_color_1_valid = gaussinput_color_1_tostep2.PopNB(color1);
                bool gaussinput_color_2_valid = gaussinput_color_2_tostep2.PopNB(color2);
                bool enable = EnableOut_to_2.PopNB(ee);
                // std::cout <<"eee11111:" << ee << std::endl;
                if (c && gaussinput_o_valid && gaussinput_color_0_valid && gaussinput_color_1_valid && gaussinput_color_2_valid && enable){
                    ac_math::ac_exp_cordic(-OUT_DATA_TYPE(0.5) * q, prob);
                    // std::cout << "pixel in  pre_calc_2:" << o << std::endl;

               // Step 4: Alpha = opacity * probability
                    OUT_DATA_TYPE alpha = prob * o;
                    tmp = alpha_reg * alpha;
                    // std::cout << "tmp:" << tmp<< std::endl;
                    // std::cout << "color:" << color0 << std::endl;
                    alpha_reg = alpha_reg - tmp;
                    if (ee){
                        alpha_reg = OUT_DATA_TYPE(0);
                    }
               // Step 5: Multiply color by alpha
                    
                    
                    // std::cout << "result in pre_calc_2_Well_1:" << result.x[0] << std::endl;
                    
                    result.x[3] = OUT_DATA_TYPE(0.0); // padding
                    EnableOut_to_3.PushNB(ee);
                    gaussinput_color_0_tostep3.PushNB(color0);
                    gaussinput_color_1_tostep3.PushNB(color1);
                    gaussinput_color_2_tostep3.PushNB(color2);
                    tmp_to_step3.PushNB(tmp);
                    // std::cout << "Cycle in PE " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                    // std::cout << "Cycle in PE after all PE finish processing " << sc_time_stamp() / sc_time(1, SC_NS) << std::endl;
                }
             
            }
        }
        #pragma hls_pipeline_init_interval 1
        void PE_CALC_3() {
            EnableOut_to_3.ResetRead();
            tmp_to_step3.ResetRead();
            gaussinput_color_0_tostep3.ResetRead();
            gaussinput_color_1_tostep3.ResetRead();
            gaussinput_color_2_tostep3.ResetRead();
            PEOutput.Reset();
            wait(4);

            #pragma hls_pipeline_init_interval 1
            while (1){
                wait();
                bool EnableOut1;
                OUT_DATA_TYPE tmp1;
                OUT_TYPE result;
                COLOR_DATA_TYPE color0;
                COLOR_DATA_TYPE color1;
                COLOR_DATA_TYPE color2;
                bool a = EnableOut_to_3.PopNB(EnableOut1);
                bool h = tmp_to_step3.PopNB(tmp1);
                bool c0 = gaussinput_color_0_tostep3.PopNB(color0);
                bool c1 = gaussinput_color_1_tostep3.PopNB(color1);
                bool c2 = gaussinput_color_2_tostep3.PopNB(color2);
                if (c0&& c1 && c2 && a && h){
                    #pragma hls_unroll
                    for (int i = 0; i < 3; i++) {
                        // std::cout << "result before pre_calc_2_Well_1:" << result.x[0] << std::endl;
                        // std::cout << "tmp * color0:" << tmp * color0<< std::endl;
                        result.x[i] += tmp1 * color0;
                        // std::cout << "result after pre_calc_2_Well_1:" << result.x[0] << std::endl;
                    }
                //    std::cout << "result" << result.x[0] << std::endl;
                std::cout << "enableOut" << EnableOut1 << std::endl;
                        if (EnableOut1) {
                            // std::cout << "AAAAAAAAAAAAAAAAAAAAAAAA";
                            PEOutput.PushNB(result);
                            #pragma hls_unroll
                            for (int i = 0; i < 3; i++) {
                                // std::cout << "result before pre_calc_2_Well_1:" << result.x[0] << std::endl;
                                // std::cout << "tmp * color0:" << tmp * color0<< std::endl;
                                result.x[i] = POS_DATA_TYPE(0);
                                // std::cout << "result after pre_calc_2_Well_1:" << result.x[0] << std::endl;
                            }
                        }
                }
            }
        }

    };











   
#endif // GSPROCESSOR_RE_H
