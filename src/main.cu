#include <iostream>
#include <chrono>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

enum gemm_type_t {
	FP32,
	FP16TC,
	FP16TC_FP16DATA
};

const char* get_mode_str(const gemm_type_t gemm_mode) {
	switch(gemm_mode) {
	case FP32:
		return "FP32";
	case FP16TC:
		return "FP16TC";
	case FP16TC_FP16DATA:
		return "FP16TC_FP16DATA";
	default:
		return "Unknown";
	}
	return "Unknown";
}

constexpr unsigned num_tests = 64;

void eval_gemm (
		const gemm_type_t gemm_type,
		const unsigned min_log_N,
		const unsigned max_log_N
		) {
	const auto mat_a_size = (1lu << (2 * max_log_N)) * (gemm_type == FP16TC_FP16DATA ? sizeof(half) : sizeof(float));
	const auto mat_b_size = (1lu << (2 * max_log_N)) * (gemm_type == FP16TC_FP16DATA ? sizeof(half) : sizeof(float));
	const auto mat_c_size = (1lu << (2 * max_log_N)) * sizeof(float);

	void *mat_a, *mat_b, *mat_c;
	cudaMalloc(&mat_a, mat_a_size);
	cudaMalloc(&mat_b, mat_b_size);
	cudaMalloc(&mat_c, mat_c_size);

	auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
	cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;
	cublasComputeType_t compute_type;
	switch (gemm_type) {
	case FP32:
		compute_type = CUBLAS_COMPUTE_32F;
		break;
	case FP16TC:
		compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
		break;
	case FP16TC_FP16DATA:
		compute_type = CUBLAS_COMPUTE_32F;
		gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
		break;
	default:
		break;
	}

	for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N++) {
		const auto N = 1lu << log_N;

		const float alpha = 1.0f;
		const float beta = 0.0f;

		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		const auto start_clock = std::chrono::system_clock::now();

		for (unsigned t = 0; t < num_tests; t++) {
			CUTF_CHECK_ERROR(cublasGemmEx(
						*cublas_handle_uptr.get(),
						CUBLAS_OP_N,
						CUBLAS_OP_N,
						N, N, N,
						&alpha,
						mat_a, (gemm_type == FP16TC_FP16DATA ? CUDA_R_16F : CUDA_R_32F), N,
						mat_b, (gemm_type == FP16TC_FP16DATA ? CUDA_R_16F : CUDA_R_32F), N,
						&beta,
						mat_c, CUDA_R_32F, N,
						compute_type,
						gemm_algo
						));
		}

		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / num_tests;

		std::printf("%s,%lu,%e,%e\n",
				get_mode_str(gemm_type),
				N,
				elapsed_time,
				2lu * N * N * N / elapsed_time * 1e-12
				);
		std::fflush(stdout);
	}

	cudaFree(mat_a);
	cudaFree(mat_b);
	cudaFree(mat_c);
}

int main(int argc, char** argv) {
	if (argc < 1 + 3) {
		std::fprintf(stderr, "Usage: %s [min_log_N] [max_log_N] [mode list: FP32 FP16TC FP16TC_FP16DATA]\n", argv[0]);
		return 1;
	}

	const auto min_log_N = std::stoul(argv[1]);
	const auto max_log_N = std::stoul(argv[2]);

	std::printf("mode,N,time,throughput_in_tflops\n");
	for (unsigned i = 3; i < argc; i++) {
		const std::string mode_str = argv[i];
		gemm_type_t gemm_type = FP32;
		if (mode_str == "FP16TC") {
			gemm_type = FP16TC;
		} else if (mode_str == "FP16TC_FP16DATA") {
			gemm_type = FP16TC_FP16DATA;
		}

		eval_gemm(gemm_type, min_log_N, max_log_N);
	}
}
