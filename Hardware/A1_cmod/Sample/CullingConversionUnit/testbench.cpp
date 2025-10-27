#define NVHLS_VERIFY_BLOCKS (CullingConversionUnit)
#include "CullingConversionUnit.h"
#include <nvhls_verify.h>
#include "nvhls_connections.h"
#include <systemc.h>
#include <nvhls_module.h>
#include <mc_connections.h>

#pragma hls_design top
class testbench : public sc_module {
public:
    sc_clock clk;
    sc_signal<bool> rst;

    Connections::Combinational<CameraParams>    CamIn;
    Connections::Combinational<Stage1a_In>      CCUInput;
    Connections::Combinational<Stage1f_Out>     CCUOutput;

    NVHLS_DESIGN(CullingConversionUnit) dut;

    SC_CTOR(testbench) : clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
                   rst("rst"),
                   CCUInput("CCUInput"),
                   CCUOutput("CCUOutput"),
                   dut("dut") {
        dut.clk(clk);
        dut.rst(rst);
        dut.CamIn(CamIn);
        dut.CCUInput(CCUInput);
        dut.CCUOutput(CCUOutput);

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
        CamIn.ResetWrite();
        CCUInput.ResetWrite();
        wait(10);
        // Simple camera (identity view, fx=fy=100, cx=cy=0), near=0.1, far=1000
        CameraParams cam; for (int r=0;r<4;r++) for (int c=0;c<4;c++) cam.view[r][c]= (r==c)? F_rot(1.0):F_rot(0.0);
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) cam.K[r][c]=F_K(0.0);
        cam.K[0][0]=F_K(100.0); cam.K[1][1]=F_K(100.0);
        cam.near_z=F_z(0.1); cam.far_z=F_z(1000.0);
        CamIn.Push(cam);

        Stage1a_In in; in.mu_w.x=F_mu(0.0); in.mu_w.y=F_mu(0.0); in.mu_w.z=F_mu(5.0); in.ID=1; in.finish=false;
        CCUInput.Push(in);
    }

    void collect() {
        CCUOutput.ResetRead();
        while (1) {
            wait();
            Stage1f_Out out = CCUOutput.Pop();
            // Expected uv = (fx*x/z+cx, fy*y/z+cy) = (0,0)
            bool pass = (out.uv.x == F_K(0)) && (out.uv.y == F_K(0)) && (out.ID==1);
            if (pass) std::cout << "[PASS] CCU projected and culled correctly." << std::endl;
            else { std::cout << "[FAIL] CCU." << std::endl; NVHLS_ASSERT_MSG(false, "CCU validation failed"); }
            sc_stop();
        }
    }
};

int sc_main(int argc, char *argv[]) {
    testbench tb("tb");
    sc_start();
    return 0;
}


