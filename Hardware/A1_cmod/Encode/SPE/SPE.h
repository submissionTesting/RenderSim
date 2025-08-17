#ifndef SRENDER_SPE_H
#define SRENDER_SPE_H

#include "SRENDERPackDef.h"
#include "DCU.h"
#include "CU.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>
#include <algorithm> 

#pragma hls_design block
class SPE : public match::Module {
    SC_HAS_PROCESS(SPE);
public:
    // Input/Output channels
    Connections::In<SPE_IN_TYPE> SPEInput;
    Connections::Out<SPE_OUT_TYPE> SPEOutput;
    Connections::Combinational<DCU_IN_TYPE> dcu_in_channels[16];
    Connections::Combinational<DCU_OUT_TYPE> dcu_out_channels[16];
    Connections::Combinational<CU_IN_TYPE> cu_ray_in_channels[16];
    Connections::Combinational<CU_OUT_TYPE> cu_ray_out_channels[16];
    Connections::Combinational<CU_IN_TYPE> cu_point_in_channels[16 * 3]; // 48 channels
    Connections::Combinational<CU_OUT_TYPE> cu_point_out_channels[16 * 3];
    // Sub-modules
    DCU* dcu_units[16];  // 16 DCU units for parallel processing
    CU* cu_ray_units[16]; // 16 CU units for ray sensitivity
    CU* cu_point_units[48]; // 48 CU units for point sensitivity

    // Constructor
    SPE(sc_module_name name) : match::Module(name),
                              SPEInput("SPEInput"),
                              SPEOutput("SPEOutput") {
        // Initialize DCU units
        for (int i = 0; i < 16; i++) {
            char dcu_name[20];
            sprintf(dcu_name, "dcu_unit_%d", i);
            dcu_units[i] = new DCU(dcu_name);
            dcu_units[i]->clk(clk);
            dcu_units[i]->rst(rst);
            dcu_units[i]->DCUInput(dcu_in_channels[i]);
            dcu_units[i]->DCUOutput(dcu_out_channels[i]);
        }
        
        // Initialize CU units for ray sensitivity
        for (int i = 0; i < 16; i++) {
            char cu_name[20];
            sprintf(cu_name, "cu_ray_unit_%d", i);
            cu_ray_units[i] = new CU(cu_name);
            cu_ray_units[i]->clk(clk);
            cu_ray_units[i]->rst(rst);
            cu_ray_units[i]->CUInput(cu_ray_in_channels[i]);
            cu_ray_units[i]->CUOutput(cu_ray_out_channels[i]);
        }
        
        // Initialize CU units for point sensitivity
        for (int i = 0; i < 48; i++) {
            char cu_name[20];
            sprintf(cu_name, "cu_point_unit_%d", i);
            cu_point_units[i] = new CU(cu_name);
            cu_point_units[i]->clk(clk);
            cu_point_units[i]->rst(rst);
            cu_point_units[i]->CUInput(cu_point_in_channels[i]);
            cu_point_units[i]->CUOutput(cu_point_out_channels[i]);
        }
        
        SC_THREAD(SPE_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }
    
    // Destructor
    ~SPE() {
        for (int i = 0; i < 16; i++) {
            delete dcu_units[i];
            delete cu_ray_units[i];
        }
        
        for (int i = 0; i < 48; i++) {
            delete cu_point_units[i];
        }
    }

    /*
     * Main processing function for sensitivity prediction
     * Implements the process described in the SRender paper:
     * 1. Color-based ray prediction using DCU and CU
     * 2. Weight-factor-based point prediction using CU
     */
    #pragma hls_pipeline_init_interval 1
    void SPE_CALC() {
        SPEInput.Reset();
        SPEOutput.Reset();
        for (int i = 0; i < 16; i++) {
            dcu_in_channels[i].ResetWrite();
            dcu_out_channels[i].ResetRead();
            cu_ray_in_channels[i].ResetWrite();
            cu_ray_out_channels[i].ResetRead();
        }
        for (int i = 0; i < 48; i++) {
            cu_point_in_channels[i].ResetWrite();
            cu_point_out_channels[i].ResetRead();
        }
        wait();

        while (1) {
            wait();
            
            // Get input data
            SPE_IN_TYPE spe_input = SPEInput.Pop();
            
            // Initialize output
            SPE_OUT_TYPE spe_output;
            
            // Arrays to store results (would be dynamically allocated in real implementation)
            static UINT16_TYPE sensitive_rays[512];  // Assuming max 512 sensitive rays
            static SensitivePointInfo sensitive_points[512];  // Assuming max 512 entries
            
            UINT16_TYPE num_sensitive_rays = 0;
            UINT16_TYPE num_sensitive_points = 0;
            
            // Stage 1: Color-based ray prediction
            // Process 3x3 regions of pixels to find sensitive rays
            for (UINT16_TYPE region_y = 0; region_y < spe_input.num_rays / 3; region_y++) {
                for (UINT16_TYPE region_x = 0; region_x < spe_input.num_rays / 3; region_x++) {
                    // For each 3x3 region, designate the middle pixel as the informative pixel
                    UINT16_TYPE region_idx = region_y * (spe_input.num_rays / 3) + region_x;
                    UINT16_TYPE informative_ray_id = region_idx * 9 + 4;  // Center pixel in 3x3 grid
                    
                    // Skip if out of bounds
                    if (informative_ray_id >= spe_input.num_rays) continue;
                    
                    RGB_TYPE informative_pixel = spe_input.rendered_pixels[informative_ray_id];
                    
                    // Mark the informative ray as sensitive
                    sensitive_rays[num_sensitive_rays++] = informative_ray_id;
                    
                    // Compare the other 8 pixels in the region with the informative pixel
                    for (int i = 0; i < 9; i++) {
                        if (i == 4) continue;  // Skip the informative pixel
                        
                        UINT16_TYPE candidate_ray_id = region_idx * 9 + i;
                        if (candidate_ray_id >= spe_input.num_rays) continue;  // Bounds check
                        
                        // Compute L1 distance between candidate and informative pixels
                        DCU_IN_TYPE dcu_in;
                        dcu_in.informative_pixel = informative_pixel;
                        dcu_in.candidate_pixel = spe_input.rendered_pixels[candidate_ray_id];
                        
                        // Select a DCU unit for this calculation
                        int dcu_idx = i % 16;  // Use modulo to distribute work among DCU units
                        dcu_in_channels[dcu_idx].Push(dcu_in);
                        DCU_OUT_TYPE dcu_out = dcu_out_channels[dcu_idx].Pop();
                        
                        // Compare L1 distance with threshold T
                        CU_IN_TYPE cu_in;
                        cu_in.input_value = dcu_out.distance;
                        cu_in.threshold = spe_input.threshold_T;
                        cu_in.id = candidate_ray_id;
                        
                        // Select a CU unit for ray sensitivity testing
                        int cu_idx = i % 16;  // Use modulo to distribute work
                        cu_ray_in_channels[cu_idx].Push(cu_in);
                        CU_OUT_TYPE cu_out = cu_ray_out_channels[cu_idx].Pop();
                        
                        // If L1 distance ≥ threshold T, mark as sensitive ray
                        if (cu_out.result == 1) {
                            sensitive_rays[num_sensitive_rays++] = candidate_ray_id;
                        }
                    }
                }
            }
            
            // Stage 2: Weight-factor-based point prediction
            // For each sensitive ray, find sensitive points
            for (UINT16_TYPE i = 0; i < num_sensitive_rays; i++) {
                UINT16_TYPE ray_id = sensitive_rays[i];
                UINT16_TYPE start_boundary = spe_input.num_points_per_ray; // Initialize to max
                UINT16_TYPE end_boundary = 0;                              // Initialize to min
                bool found_sensitive = false;
                
                // Traverse points from beginning to end
                for (UINT16_TYPE j = 0; j < spe_input.num_points_per_ray; j++) {
                    UINT16_TYPE point_idx = ray_id * spe_input.num_points_per_ray + j;
                    
                    // Compare weight factor with threshold D
                    CU_IN_TYPE cu_in;
                    cu_in.input_value = spe_input.weight_factors[point_idx];
                    cu_in.threshold = spe_input.threshold_D;
                    cu_in.id = j;
                    
                    // Select a CU unit for point sensitivity testing
                    int cu_idx = j % 48;  // Use modulo to distribute work
                    cu_point_in_channels[cu_idx].Push(cu_in);
                    CU_OUT_TYPE cu_out = cu_point_out_channels[cu_idx].Pop();
                    
                    // If weight factor ≥ threshold D, update boundaries
                    if (cu_out.result == 1) {
                        found_sensitive = true;
                        start_boundary = std::min(start_boundary, j);
                        end_boundary = std::max(end_boundary, j);
                    }
                }
                
                // If sensitive points were found, record their boundaries
                if (found_sensitive) {
                    sensitive_points[num_sensitive_points].ray_id = ray_id;
                    sensitive_points[num_sensitive_points].start_point_id = start_boundary;
                    sensitive_points[num_sensitive_points].end_point_id = end_boundary;
                    num_sensitive_points++;
                }
            }
            
            // Set output data
            for (int i = 0; i < num_sensitive_rays; ++i) {
                spe_output.sensitive_rays[i] = sensitive_rays[i];
            }
            spe_output.num_sensitive_rays = num_sensitive_rays;

            for (int i = 0; i < num_sensitive_points; ++i) {
                spe_output.sensitive_points[i] = sensitive_points[i];
            }
            spe_output.num_sensitive_points = num_sensitive_points;
            
            // Push output
            SPEOutput.Push(spe_output);
        }
    }
};
#endif // SRENDER_SPE_H