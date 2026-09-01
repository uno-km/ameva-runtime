#include "../doctor/probe_stages.h"
#include "../c_api/ameva_vulkan_c_api.h"
#include "../quirks/mali_quirks.h"
#include "../quirks/adreno_quirks.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

void PrintUsage(const char* prog) {
    std::cout << "AMEVA Vulkan Hardware Diagnostic & Runtime Tool (Native C++ CLI)\n"
              << "Usage: " << prog << " [command]\n\n"
              << "Commands:\n"
              << "  doctor      Run 12-stage Vulkan hardware validation (V0-V11)\n"
              << "  benchmark   Run native GEMM matrix throughput benchmark\n"
              << "  version     Print runtime version information\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "doctor") {
        ameva::doctor::ProbeSuite suite;
        ameva::doctor::DiagnosticReport report = suite.RunFullDiagnostic(true);
        suite.SaveState(report);
        return report.overall_success ? 0 : 1;
    }
    else if (cmd == "benchmark") {
        std::cout << "\n============================================================\n"
                  << "  AMEVA-Vulkan-Runtime: Native Micro-GEMM Benchmark         \n"
                  << "============================================================\n";

        const int M = 256, K = 256, N = 256;
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 0.5f);
        std::vector<float> C(M * N, 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
        int res = ameva_matmul_f32(A.data(), B.data(), C.data(), M, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ops = 2.0 * M * K * N;
        double gflops = (elapsed_ms > 0) ? (ops / (elapsed_ms * 1e6)) : 0.0;

        std::cout << "  - Matrix Dimension: " << M << " x " << K << " x " << N << "\n"
                  << "  - Execution Status: " << (res == 0 ? "SUCCESS" : "FAILED") << "\n"
                  << "  - Elapsed Time:     " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n"
                  << "  - Throughput:       " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n"
                  << "============================================================\n\n";
        return res == 0 ? 0 : 1;
    }
    else if (cmd == "version") {
        std::cout << "AMEVA-Vulkan-Runtime Native HAL v" << ameva_get_version() << std::endl;
        return 0;
    }
    else {
        std::cerr << "Unknown command: " << cmd << "\n\n";
        PrintUsage(argv[0]);
        return 1;
    }
}
