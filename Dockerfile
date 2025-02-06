# Start from ROCm capable base image
FROM rocm/pytorch-training:v25.2
WORKDIR /workspace

# Copy resources from FluxBenchmark and install requirements
COPY . FluxBenchmark
RUN cd FluxBenchmark && \
    pip3 install --no-cache-dir --upgrade pip packaging && \
    pip3 install --no-cache-dir -r requirements.txt

# Setup HF's home
ENV HF_HOME /workspace/huggingface

# Set ENV variables that maximize perf:
# Enable hipBLASLt as the backend for optimized GEMM (General Matrix Multiply) operations in rocBLAS
ENV ROCBLAS_USE_HIPBLASLT 1
# Enable CUDA Lightweight (LT) optimizations for the addmm operation in PyTorch
ENV DISABLE_ADDMM_CUDA_LT 0
# Force the HIP runtime to place kernel arguments directly in device memory instead of host memory.
ENV HIP_FORCE_DEV_KERNARG 1
# Set NCCL operations to default-priority streams.
ENV TORCH_NCCL_HIGH_PRIORITY 0
# Configure the maximum number of hardware queues per GPU in the HIP runtime, enabling better parallelism.
ENV GPU_MAX_HW_QUEUES 8

# Set workdir to FluxBenchmark
WORKDIR /workspace/FluxBenchmark