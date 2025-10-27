#ifndef GSARCH_CULLINGCONVERSIONUNIT_H
#define GSARCH_CULLINGCONVERSIONUNIT_H

#include "../include/GSARCHPackDef.h"
#include <ac_channel.h>
#include <nvhls_connections.h>
#include <ac_math/ac_reciprocal_pwl.h>

// CCU: culling (near/far, screen) + projection (Stage1a/b subset)
#pragma hls_design block
class CullingConversionUnit : public match::Module {
    SC_HAS_PROCESS(CullingConversionUnit);
public:
    Connections::In<CameraParams>    CamIn;
    Connections::In<Stage1a_In>      CCUInput;
    Connections::Out<Stage1f_Out>    CCUOutput;

    CullingConversionUnit(sc_module_name name) : match::Module(name),
                                                 CamIn("CamIn"),
                                                 CCUInput("CCUInput"),
                                                 CCUOutput("CCUOutput") {
        SC_THREAD(CCU_CALC);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    void CCU_CALC() {
        CamIn.Reset();
        CCUInput.Reset();
        CCUOutput.Reset();
        wait();
        CameraParams cam = CamIn.Pop();
        F_rot Vloc[4][4]; F_K Kloc[3][3]; F_z near_z = cam.near_z; F_z far_z = cam.far_z;
        for (int r=0;r<4;r++) for (int c=0;c<4;c++) Vloc[r][c]=cam.view[r][c];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) Kloc[r][c]=cam.K[r][c];

        #pragma hls_pipeline_init_interval 1
        while (1) {
            wait();
            Stage1a_In in = CCUInput.Pop();
            // View transform (3x4 * mu_w)
            F_acc s0 = F_acc(Vloc[0][0])*F_acc(in.mu_w.x) + F_acc(Vloc[0][1])*F_acc(in.mu_w.y) + F_acc(Vloc[0][2])*F_acc(in.mu_w.z) + F_acc(Vloc[0][3]);
            F_acc s1 = F_acc(Vloc[1][0])*F_acc(in.mu_w.x) + F_acc(Vloc[1][1])*F_acc(in.mu_w.y) + F_acc(Vloc[1][2])*F_acc(in.mu_w.z) + F_acc(Vloc[1][3]);
            F_acc s2 = F_acc(Vloc[2][0])*F_acc(in.mu_w.x) + F_acc(Vloc[2][1])*F_acc(in.mu_w.y) + F_acc(Vloc[2][2])*F_acc(in.mu_w.z) + F_acc(Vloc[2][3]);

            // Near/far cull on s2
            F_cmp s2c = F_cmp(s2);
            bool depth_ok = (s2c >= F_cmp(near_z)) && (s2c <= F_cmp(far_z));
            if (depth_ok) {
                // invz = 1/z
                F_mu z_mu = F_mu(s2);
                F_mu eps_mu = F_mu(1e-6);
                F_mu z = (z_mu > eps_mu) ? z_mu : eps_mu;
                F_invz invz; ac_math::ac_reciprocal_pwl(z, invz);

                // Project to uv
                F_acc xn = F_acc(s0) * F_acc(invz);
                F_acc yn = F_acc(s1) * F_acc(invz);
                F_acc uvx_acc = F_acc(Kloc[0][0]) * xn + F_acc(Kloc[0][2]);
                F_acc uvy_acc = F_acc(Kloc[1][1]) * yn + F_acc(Kloc[1][2]);
                float2 uv; uv.x = F_K(uvx_acc); uv.y = F_K(uvy_acc);
                // Screen cull
                bool on = (uv.x >= F_K(0)) && (uv.x <= F_K(SCREEN_WIDTH)) && (uv.y >= F_K(0)) && (uv.y <= F_K(SCREEN_HEIGHT));
                if (on) {
                    // Minimal downstream packaging (reuse Stage1e/f outputs where possible)
                    Stage1f_Out o; o.ID = in.ID; o.uv = uv; o.finish = in.finish; o.radius_px = F_K(1.0);
                    o.cov2d[0][0] = F_cov2d(1.0); o.cov2d[0][1] = F_cov2d(0.0); o.cov2d[1][0] = F_cov2d(0.0); o.cov2d[1][1] = F_cov2d(1.0);
                    CCUOutput.PushNB(o);
                }
            }
        }
    }
};

#endif // GSARCH_CULLINGCONVERSIONUNIT_H


