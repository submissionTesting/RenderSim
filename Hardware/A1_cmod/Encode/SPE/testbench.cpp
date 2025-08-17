#define NVHLS_VERIFY_BLOCKS (SPE)
#include "SPE.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include "ac_sysc_trace.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<SPE_IN_TYPE> SPEInput;
    Connections::Combinational<SPE_OUT_TYPE> SPEOutput;

    NVHLS_DESIGN(SPE) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   SPEInput("SPEInput"),
                   SPEOutput("SPEOutput"),
                   dut("dut") {

        sc_object_tracer<sc_clock> trace_clk(clk);

        dut.clk(clk);
        dut.rst(rst);
        dut.SPEInput(SPEInput);
        dut.SPEOutput(SPEOutput);

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
        SPEInput.ResetWrite();
        wait(10);

        // Setup a simple test case with a 3x3 image grid (9 rays)
        // and 4 points per ray
        const int NUM_RAYS = 9;
        const int NUM_POINTS_PER_RAY = 4;
        
        // Allocate memory for rendered pixels and weight factors
        RGB_TYPE rendered_pixels[NUM_RAYS];
        FP16_TYPE weight_factors[NUM_RAYS * NUM_POINTS_PER_RAY];
        
        // Initialize rendered pixels
        // Create a pattern where pixels at positions 0, 4, and 8 are similar
        // while pixels at positions 1, 2, 3, 5, 6, 7 are different
        
        // Pixel 0 (top-left)
        rendered_pixels[0].r = FP16_TYPE(100.0);
        rendered_pixels[0].g = FP16_TYPE(100.0);
        rendered_pixels[0].b = FP16_TYPE(100.0);
        
        // Pixel 1 (top-center)
        rendered_pixels[1].r = FP16_TYPE(150.0);
        rendered_pixels[1].g = FP16_TYPE(150.0);
        rendered_pixels[1].b = FP16_TYPE(150.0);
        
        // Pixel 2 (top-right)
        rendered_pixels[2].r = FP16_TYPE(200.0);
        rendered_pixels[2].g = FP16_TYPE(200.0);
        rendered_pixels[2].b = FP16_TYPE(200.0);
        
        // Pixel 3 (middle-left)
        rendered_pixels[3].r = FP16_TYPE(110.0);
        rendered_pixels[3].g = FP16_TYPE(110.0);
        rendered_pixels[3].b = FP16_TYPE(110.0);
        
        // Pixel 4 (middle-center) - informative pixel
        rendered_pixels[4].r = FP16_TYPE(100.0);
        rendered_pixels[4].g = FP16_TYPE(100.0);
        rendered_pixels[4].b = FP16_TYPE(100.0);
        
        // Pixel 5 (middle-right)
        rendered_pixels[5].r = FP16_TYPE(90.0);
        rendered_pixels[5].g = FP16_TYPE(90.0);
        rendered_pixels[5].b = FP16_TYPE(90.0);
        
        // Pixel 6 (bottom-left)
        rendered_pixels[6].r = FP16_TYPE(180.0);
        rendered_pixels[6].g = FP16_TYPE(180.0);
        rendered_pixels[6].b = FP16_TYPE(180.0);
        
        // Pixel 7 (bottom-center)
        rendered_pixels[7].r = FP16_TYPE(120.0);
        rendered_pixels[7].g = FP16_TYPE(120.0);
        rendered_pixels[7].b = FP16_TYPE(120.0);
        
        // Pixel 8 (bottom-right)
        rendered_pixels[8].r = FP16_TYPE(105.0);
        rendered_pixels[8].g = FP16_TYPE(105.0);
        rendered_pixels[8].b = FP16_TYPE(105.0);
        
        // Initialize weight factors
        // For each ray, we'll set a pattern where the first and last points have low weights
        // and the middle points have high weights
        // This should make the middle points sensitive
        for (UINT16_TYPE ray = 0; ray < NUM_RAYS; ray++) {
            for (UINT16_TYPE point = 0; point < NUM_POINTS_PER_RAY; point++) {
                UINT16_TYPE idx = ray * NUM_POINTS_PER_RAY + point;
                
                if (point == 0 || point == NUM_POINTS_PER_RAY - 1) {
                    // First and last points have low weight factors
                    weight_factors[idx] = FP16_TYPE(0.005); // Below threshold D
                } else {
                    // Middle points have high weight factors
                    weight_factors[idx] = FP16_TYPE(0.05); // Above threshold D
                }
            }
        }

        // Prepare the input data structure
        SPE_IN_TYPE spe_input;
        // Input assignment example
        for (UINT16_TYPE i = 0; i < NUM_RAYS; ++i) {
            spe_input.rendered_pixels[i] = rendered_pixels[i];
        }
        for (UINT16_TYPE i = 0; i < NUM_RAYS * NUM_POINTS_PER_RAY; ++i) {
            spe_input.weight_factors[i] = weight_factors[i];
        }
        spe_input.num_rays = NUM_RAYS;
        spe_input.num_points_per_ray = NUM_POINTS_PER_RAY;
        spe_input.threshold_T = FP16_TYPE(20.0);  // Threshold for ray sensitivity
        spe_input.threshold_D = FP16_TYPE(0.01);  // Threshold for point sensitivity
        
        cout << "\n===== SPE Testbench: Test Case Setup =====\n";
        cout << "Number of rays: " << NUM_RAYS << endl;
        cout << "Number of points per ray: " << NUM_POINTS_PER_RAY << endl;
        cout << "Threshold T (ray sensitivity): " << spe_input.threshold_T << endl;
        cout << "Threshold D (point sensitivity): " << spe_input.threshold_D << endl;
        
        cout << "\nRendered Pixels (RGB values):\n";
        for (int i = 0; i < NUM_RAYS; i++) {
            cout << "Pixel " << i << ": ("
                 << rendered_pixels[i].r << ", "
                 << rendered_pixels[i].g << ", "
                 << rendered_pixels[i].b << ")" << endl;
        }
        
        cout << "\nExpected Ray Sensitivity:\n";
        cout << "Pixel 4 is the informative pixel.\n";
        cout << "Pixels 0 and 8 are similar to Pixel 4 (L1 distance < 20) -> Insensitive\n";
        cout << "Pixels 1, 2, 3, 5, 6, 7 are different from Pixel 4 (L1 distance > 20) -> Sensitive\n";
        
        cout << "\nExpected Point Sensitivity:\n";
        cout << "For each sensitive ray:\n";
        cout << "Points 1 and 2 have weight factor 0.05 (> threshold 0.01) -> Sensitive\n";
        cout << "Points 0 and 3 have weight factor 0.005 (< threshold 0.01) -> Insensitive\n";
        
        // Send input to SPE
        SPEInput.Push(spe_input);
        
        wait(100); // Wait sufficient time for SPE to process
    }

    void collect() {
        SPEOutput.ResetRead();
        wait(20);  // Wait for reset and initialization

        // Wait for SPE to finish processing
        wait(100);
        
        // Get output
        SPE_OUT_TYPE spe_output = SPEOutput.Pop();
        
        cout << "\n===== SPE Testbench: Test Results =====\n";
        
        cout << "\nSensitive Rays Detected: " << spe_output.num_sensitive_rays << endl;
        cout << "Ray IDs: ";
        for (UINT16_TYPE i = 0; i < spe_output.num_sensitive_rays; i++) {
            cout << spe_output.sensitive_rays[i] << " ";
        }
        cout << endl;
        
        cout << "\nSensitive Points Detected: " << spe_output.num_sensitive_points << endl;
        cout << "Ray ID\tStart Point\tEnd Point" << endl;
        for (UINT16_TYPE i = 0; i < spe_output.num_sensitive_points; i++) {
            cout << spe_output.sensitive_points[i].ray_id << "\t"
                 << spe_output.sensitive_points[i].start_point_id << "\t\t"
                 << spe_output.sensitive_points[i].end_point_id << endl;
        }
        
        cout << "\n===== SPE Testbench: Test Completed =====\n";
        
        sc_stop();
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}