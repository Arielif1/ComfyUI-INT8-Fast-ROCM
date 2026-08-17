# rocblas_int8.py — rocBLAS-backed W8A8 int8 GEMM backend for RDNA2 (gfx103x).
# Drop-in replacement for ComfyUI-INT8-Fast-ROCM's triton_int8_linear /
# triton_int8_linear_per_row: same signatures, same numerics (int8 GEMM accumulation
# is exact integer arithmetic -> bit-identical int32; dequant mirrors the pack's
# reference formulas). Uses hipblasGemmEx -> rocblas gfx1032 I8I_HPA/4xi8I_HPA (DP4a)
# kernels, bypassing hipblasLt and triton's FMA-emulated tl.dot.
#
# WHY THIS EXISTS (2026-08-16 session 3, RX 6600 / gfx1032):
#   - The RX 6600 HAS native int8 hardware: v_dot4_i32_i8 (DP4a) runs at exactly
#     2x fp16, v_dot8_i32_i4 at 4x fp16 (measured, see bench_ext.py).
#   - ComfyUI-INT8-Fast-ROCM's own int8 kernels are Triton tl.dot, which on gfx103x
#     compiles to FMA emulation -> measured ~4x SLOWER than fp16. Not usable.
#   - torch._int_mm crashes (HIPBLAS_STATUS_INVALID_VALUE): the rocm-sdk pack only
#     ships hipblaslt gfx1100 kernels; gfx1032 was never supported by hipBLASLt in
#     ANY ROCm version (dead end, see hipblaslt-int8-mission.md).
#   - rocBLAS DOES ship gfx1032 int8 kernels (I8I_HPA / 4xi8I_HPA = DP4a). Calling
#     them directly via hipblasGemmEx gives measured 1.24-1.39x over fp16 at MLP
#     shapes (4096x2048x8192 etc.).
#
# BUILD / LOAD (two paths):
#   - First build: run rocblas_int8_quick.bat (calls vcvars64.bat for MSVC, sets
#     HSA_OVERRIDE_GFX_VERSION=10.3.0). The .pyd lands in _torch_ext/rocblas_int8_gemm/.
#   - Every subsequent load: _direct_import() loads the prebuilt .pyd via importlib.
#     This is REQUIRED inside ComfyUI: torch's load_inline regenerates the ninja file
#     on EVERY call (its versioner hash is per-process) and that path runs `where cl`,
#     so it demands MSVC on PATH every time. Direct import needs zero toolchain.
#   - NOTE: import torch BEFORE the .pyd so torch's DLLs are on the search path
#     (the extension links against torch_hip.dll etc.).
import os
os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"C:\Users\ariel\Documents\HermesProjects\anima-lora-training\_torch_ext")
import torch
import torch.utils.cpp_extension as cpp_ext

DEVEL = r"C:\VariousPrograms\cu\comfyui-rocm\python_env\Lib\site-packages\_rocm_sdk_devel"

_CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <math.h>

static hipblasHandle_t g_handle = nullptr;
static void ensure_handle() {
    if (!g_handle) hipblasCreate(&g_handle);
}

// C = A @ B : A [M,K] row-major, B [K,N] row-major, C [M,N] int32.
// No-copy trick: row-major buffers == transposed col-major views; compute C^T = B^T @ A^T.
extern "C" int int8_gemm(int m, int n, int k,
                         const int8_t* A, const int8_t* B, int32_t* C) {
    ensure_handle();
    const int32_t alpha = 1, beta = 0;
    hipblasStatus_t st = hipblasGemmEx(
        g_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        n, m, k,
        &alpha,
        B, HIPBLAS_R_8I, n,
        A, HIPBLAS_R_8I, k,
        &beta,
        C, HIPBLAS_R_32I, n,
        HIPBLAS_COMPUTE_32I,
        HIPBLAS_GEMM_DEFAULT);
    return (int)st;
}

// C = A @ W^T : A [M,K] row-major, W [N,K] row-major (linear weight), C [M,N] int32.
// C^T = W @ A^T. gemm(M_g=N, N_g=M, K_g=K) with transa=OP_T (operand = W, ld=K),
// B_g = A (col-major KxM, ld=K), C_g (NxM col-major) = C^T, ld=N.
extern "C" int int8_gemm_nt(int m, int n, int k,
                            const int8_t* A, const int8_t* W, int32_t* C) {
    ensure_handle();
    const int32_t alpha = 1, beta = 0;
    hipblasStatus_t st = hipblasGemmEx(
        g_handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
        n, m, k,
        &alpha,
        W, HIPBLAS_R_8I, k,
        A, HIPBLAS_R_8I, k,
        &beta,
        C, HIPBLAS_R_32I, n,
        HIPBLAS_COMPUTE_32I,
        HIPBLAS_GEMM_DEFAULT);
    return (int)st;
}

// ---- fused rowwise quantize: x fp16 [M,K] -> y int8 [M,K], s fp32 [M] ----
// Bit-matches the node pack's triton quantizer: max computed on exact fp16 values,
// scale = max(max|x|/127, 1e-30) in fp32, q = floor(x/scale + 0.5) clamp [-128,127].
__global__ void quantize_rowwise_kernel(const __half* __restrict__ x,
                                        int8_t* __restrict__ y,
                                        float* __restrict__ s,
                                        int M, int K) {
    const int row = __builtin_amdgcn_workgroup_id_x();
    const __half* xr = x + (size_t)row * K;
    int8_t* yr = y + (size_t)row * K;
    const int tid = __builtin_amdgcn_workitem_id_x();
    const int nthreads = __builtin_amdgcn_workgroup_size_x();
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float m = 0.f;
    for (int i = tid; i < K; i += nthreads) {
        float v = __half2float(xr[i]);
        m = fmaxf(m, fabsf(v));
    }
    __shared__ float smem[32];
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down(m, o));
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < (nthreads >> 5)) ? smem[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down(m, o));
        if (lane == 0) smem[0] = fmaxf(m / 127.0f, 1e-30f);
    }
    __syncthreads();
    const float scale = smem[0];
    if (tid == 0) s[row] = scale;
    for (int i = tid; i < K; i += nthreads) {
        float v = __half2float(xr[i]);
        float q = floorf(v / scale + 0.5f);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        yr[i] = (int8_t)q;
    }
}

// ---- fused dequant epilogue: out = (acc * (x_scale[r] * w_scale[c]) [+ bias[c]]) -> fp16 ----
__global__ void dequant_kernel(const int32_t* __restrict__ acc,
                               const float* __restrict__ x_scale,
                               const float* __restrict__ w_scale,
                               const __half* __restrict__ bias,
                               __half* __restrict__ out,
                               int M, int N, int has_bias) {
    const int idx = __builtin_amdgcn_workitem_id_x() +
                    __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x();
    if (idx >= M * N) return;
    const int r = idx / N, c = idx - r * N;
    float v = (float)acc[idx] * (x_scale[r] * w_scale[c]);
    if (has_bias) v += __half2float(bias[c]);
    out[idx] = __float2half(v);
}

extern "C" void quantize_rowwise_launch(const void* x, void* y, void* s, int M, int K) {
    quantize_rowwise_kernel<<<M, 256>>>((const __half*)x, (int8_t*)y, (float*)s, M, K);
}

extern "C" void dequant_launch(const void* acc, const void* x_scale, const void* w_scale,
                               const void* bias, void* out, int M, int N, int has_bias) {
    const long total = (long)M * N;
    const int blocks = (int)((total + 255) / 256);
    dequant_kernel<<<blocks, 256>>>((const int32_t*)acc, (const float*)x_scale,
                                    (const float*)w_scale, (const __half*)bias,
                                    (__half*)out, M, N, has_bias);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>
extern "C" int int8_gemm(int m, int n, int k, const int8_t* A, const int8_t* B, int32_t* C);
extern "C" int int8_gemm_nt(int m, int n, int k, const int8_t* A, const int8_t* W, int32_t* C);
extern "C" void quantize_rowwise_launch(const void* x, void* y, void* s, int M, int K);
extern "C" void dequant_launch(const void* acc, const void* x_scale, const void* w_scale,
                               const void* bias, void* out, int M, int N, int has_bias);

torch::Tensor int8_gemm_t(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuda tensors required");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "int8 required");
    int m = a.size(0), k = a.size(1), n = b.size(1);
    TORCH_CHECK(b.size(0) == k, "inner dim mismatch");
    TORCH_CHECK(k % 4 == 0, "k must be a multiple of 4 (DP4a)");
    auto c = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    int st = int8_gemm(m, n, k, a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<int32_t>());
    TORCH_CHECK(st == 0, "hipblasGemmEx failed, status=", st);
    return c;
}

torch::Tensor int8_gemm_nt_t(torch::Tensor a, torch::Tensor w) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda(), "cuda tensors required");
    TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8, "int8 required");
    int m = a.size(0), k = a.size(1), n = w.size(0);
    TORCH_CHECK(w.size(1) == k, "inner dim mismatch");
    TORCH_CHECK(k % 4 == 0, "k must be a multiple of 4 (DP4a)");
    auto c = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    int st = int8_gemm_nt(m, n, k, a.data_ptr<int8_t>(), w.data_ptr<int8_t>(), c.data_ptr<int32_t>());
    TORCH_CHECK(st == 0, "hipblasGemmEx failed, status=", st);
    return c;
}

// x fp16 [M,K] -> y int8 [M,K], s fp32 [M,1]
void quantize_rowwise_t(torch::Tensor x, torch::Tensor y, torch::Tensor s) {
    quantize_rowwise_launch(x.data_ptr(), y.data_ptr(), s.data_ptr(), x.size(0), x.size(1));
}

// acc int32 [M,N] * (x_scale[M,1] * w_scale[N]) [+ bias[N]] -> out fp16 [M,N]
void dequant_t(torch::Tensor acc, torch::Tensor x_scale, torch::Tensor w_scale,
               torch::Tensor bias, torch::Tensor out) {
    int has_bias = bias.numel() > 0 ? 1 : 0;
    dequant_launch(acc.data_ptr(), x_scale.data_ptr(), w_scale.data_ptr(),
                   has_bias ? bias.data_ptr() : nullptr, out.data_ptr(),
                   acc.size(0), acc.size(1), has_bias);
}
"""

_ext = None

# --- OP_N support (pre-transposed weights) ---
# 2026-08-17: measured on gfx1032 real anima shape mix — OP_N (no-copy B^T@A^T
# trick via pre-transposed [K,N] weight) is 0.80-0.94x of the OP_T variant
# (op_n_probe.py). Toggle: ROCM_INT8_OPN=0 restores the old OP_T path.
# NOTE 2026-08-17: an initial version cached wT keyed on (data_ptr, shape); a
# full-model trace PROVED the pack reuses the same storage addresses for
# different layers' weights (alias collisions -> stale wT -> garbage output).
# Per-call transpose is correct but the full-model 10-step bench regressed
# (6.03 vs 5.52 s/it) — the transpose kernels/allocations disrupt the pipeline.
# OP_T remains the default; OP_N stays available via ROCM_INT8_OPN=1 for
# experiments (microbench-proven faster per-shape, so revisit post-fusion).
_USE_OPN = os.environ.get("ROCM_INT8_OPN", "0").strip().lower() in ("1", "true", "on", "yes")


def _pre_transposed(weight: torch.Tensor):
    """Transpose a [N,K] int8 linear weight to [K,N] for the OP_N GEMM path."""
    return weight.T.contiguous()


def _direct_import():
    """Import the prebuilt .pyd directly (no MSVC/ninja needed).

    torch's load_inline on Windows always regenerates the ninja file, which
    calls `where cl` even for cached builds (versioner is per-process), so it
    requires MSVC on PATH on EVERY load -- unacceptable inside ComfyUI.
    The .pyd is a plain pybind11 module; importlib loads it directly.
    """
    import importlib.util
    pyd = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", ""), "rocblas_int8_gemm", "rocblas_int8_gemm.pyd")
    if not os.path.exists(pyd):
        return None
    spec = importlib.util.spec_from_file_location("rocblas_int8_gemm", pyd)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ext():
    global _ext
    if _ext is None:
        _ext = _direct_import()
        if _ext is not None:
            return _ext
        _ext = cpp_ext.load_inline(
            name="rocblas_int8_gemm", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
            functions=["int8_gemm_t", "int8_gemm_nt_t", "quantize_rowwise_t", "dequant_t"], verbose=False,
            extra_cuda_cflags=["-nogpulib", f"-I{DEVEL}\\include"],
            extra_ldflags=[f"{DEVEL}\\lib\\hipblas.lib"],
        )
    return _ext


def quantize_rowwise(x: torch.Tensor):
    """Fused kernel: x fp16 [M,K] -> (int8 [M,K], scale fp32 [M,1]).
    Bit-matches the pack's triton quantizer. bf16/fp32 inputs fall back to torch ops."""
    if x.dtype != torch.float16:
        return quantize_rowwise_torch(x)
    ext = _load_ext()
    M, K = x.shape
    y = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    ext.quantize_rowwise_t(x.contiguous(), y, s)
    return y, s


def quantize_rowwise_torch(x: torch.Tensor):
    """Reference torch implementation (used for non-fp16 inputs + verification)."""
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(dim=-1, keepdim=True) / 127.0, min=1e-30)
    q = torch.clamp(torch.floor(xf / scale + 0.5), -128.0, 127.0).to(torch.int8)
    return q, scale.to(torch.float32)


def _dequant(acc, x_scale, w_scale, bias, compute_dtype):
    M, N = acc.shape
    if compute_dtype != torch.float16:
        # HIP kernel writes __float2half (fp16 bits) -- only safe for fp16 out.
        # bf16/fp32: do the dequant in torch (elementwise, memory-bound, cheap).
        out = acc.float() * (x_scale * w_scale.reshape(1, -1))
        if bias is not None:
            out = out + bias.float()
        return out.to(compute_dtype)
    ext = _load_ext()
    out = torch.empty((M, N), device=acc.device, dtype=compute_dtype)
    ext.dequant_t(acc, x_scale, w_scale, bias if bias is not None else torch.empty(0, device=acc.device), out)
    return out


def int8_linear(x, weight, weight_scale, bias=None, compute_dtype=torch.float16):
    """Drop-in for triton_int8_linear (per-channel weight scale [N] or scalar)."""
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    N = weight.shape[0]
    x_int8, x_scale = quantize_rowwise(x_2d)
    if _USE_OPN:
        acc = _load_ext().int8_gemm_t(x_int8, _pre_transposed(weight))  # [M, N] int32, exact
    else:
        acc = _load_ext().int8_gemm_nt_t(x_int8, weight)  # [M, N] int32, exact
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    ws = weight_scale.to(x.device).to(torch.float32).reshape(-1)  # [N] or [1]
    if ws.numel() == 1 and N > 1:
        # dequant kernel indexes w_scale[c] for c in [0,N) -- materialize the broadcast
        ws = ws.expand(N).contiguous()
    out = _dequant(acc, x_scale, ws, bias, compute_dtype)
    return out.reshape(x_shape_orig[:-1] + (N,))


def int8_linear_per_row(x, weight, weight_scale, bias=None, compute_dtype=torch.float16):
    """Drop-in for triton_int8_linear_per_row (per-row weight scale [N,1] or [N])."""
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    N = weight.shape[0]
    x_int8, x_scale = quantize_rowwise(x_2d)
    if _USE_OPN:
        acc = _load_ext().int8_gemm_t(x_int8, _pre_transposed(weight))  # [M, N] int32, exact
    else:
        acc = _load_ext().int8_gemm_nt_t(x_int8, weight)  # [M, N] int32, exact
    ws = weight_scale.to(x.device).to(torch.float32).reshape(-1)  # [N]
    out = _dequant(acc, x_scale, ws, bias, compute_dtype)
    return out.reshape(x_shape_orig[:-1] + (N,))


if __name__ == "__main__":
    import time, importlib.util, sys

    t0 = time.time()
    ext = _load_ext()
    print(f"compile/load OK in {time.time()-t0:.1f}s | device {torch.cuda.get_device_name(0)}")

    # --- correctness of nt variant vs fp64 ---
    torch.manual_seed(0)
    for (m, n, k) in [(2048, 2048, 2048), (4096, 2048, 8192), (7, 5, 1024)]:
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        ref = (a.float().double() @ w.float().double().T).long()
        got = ext.int8_gemm_nt_t(a, w).long()
        ok = torch.equal(got, ref)
        # also the plain variant
        b = torch.randint(-127, 128, (k, n), device="cuda", dtype=torch.int8)
        ok2 = torch.equal(ext.int8_gemm_t(a, b).long(), (a.float().double() @ b.float().double()).long())
        print(f"  nt {m}x{n}x{k} exact={ok} | plain exact={ok2}")

    # --- quantize kernel vs torch reference (bit-equality) ---
    torch.manual_seed(2)
    for (m, k) in [(2048, 8192), (7, 1024), (4096, 2048)]:
        x = (torch.randn(m, k, device="cuda") * 3.0).half()
        q1, s1 = quantize_rowwise(x)
        q2, s2 = quantize_rowwise_torch(x)
        print(f"  quant {m}x{k}: int8 equal={torch.equal(q1, q2)} scale equal={torch.equal(s1, s2)}")

    # --- full linear vs exact fp64 reference ---
    torch.manual_seed(3)
    for (m, n, k) in [(2048, 2048, 2048), (4096, 2048, 8192)]:
        x = (torch.randn(m, k, device="cuda") * 2.0).half()
        w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        ws = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001)
        bias = (torch.randn(n, device="cuda") * 0.1).half()
        y = int8_linear(x, w, ws, bias)
        xi, xs = quantize_rowwise_torch(x)
        ref = ((xi.float().double() @ w.float().double().T) * (xs.double() * ws.double().reshape(1, -1)) + bias.double()).half()
        same = torch.equal(y, ref)
        maxdiff = (y.float() - ref.float()).abs().max().item()
        print(f"  linear {m}x{n}x{k}: bit-equal vs fp64-ref={same} (maxdiff {maxdiff:.3e})")

    # --- vs the node pack's triton functions (bit-identity + speed) ---
    try:
        spec = importlib.util.spec_from_file_location(
            "int8_fused_kernel",
            r"C:\VariousPrograms\cu\comfyui-rocm\custom_nodes\ComfyUI-INT8-Fast-ROCM\int8_fused_kernel.py")
        tfk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tfk)
        print("node-pack triton kernels imported OK")
    except Exception as e:
        print("triton import failed:", e)
        tfk = None

    def compare_and_time(name, m, n, k, per_row=False):
        torch.manual_seed(1)
        x = (torch.randn(m, k, device="cuda") * 2.0).half()
        w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        ws = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001)
        if per_row:
            ws = ws.reshape(n, 1)
        bias = (torch.randn(n, device="cuda") * 0.1).half()
        f_roc = int8_linear_per_row if per_row else int8_linear
        f_tri = tfk.triton_int8_linear_per_row if per_row else tfk.triton_int8_linear
        with torch.no_grad():
            y_roc = f_roc(x, w, ws, bias)
            # --- isolation: GEMM-only (OP_T variant) vs pre-transposed OP_N ---
            x_int8, x_scale = quantize_rowwise(x.reshape(-1, k))
            wT = w.T.contiguous()  # [K, N] — one-time cost in practice (static weights)
            for _ in range(3): ext.int8_gemm_nt_t(x_int8, w)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): ext.int8_gemm_nt_t(x_int8, w)
            torch.cuda.synchronize()
            dt_nt = (time.perf_counter() - t0) / 10
            for _ in range(3): ext.int8_gemm_t(x_int8, wT)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): ext.int8_gemm_t(x_int8, wT)
            torch.cuda.synchronize()
            dt_t = (time.perf_counter() - t0) / 10
            # quantize-only
            for _ in range(3): quantize_rowwise(x.reshape(-1, k))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): quantize_rowwise(x.reshape(-1, k))
            torch.cuda.synchronize()
            dt_q = (time.perf_counter() - t0) / 10
            print(f"  gemm OP_T {dt_nt*1000:6.2f} ms | gemm OP_N(pre-T) {dt_t*1000:6.2f} ms | quant {dt_q*1000:5.2f} ms")
            for _ in range(3): f_roc(x, w, ws, bias)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): f_roc(x, w, ws, bias)
            torch.cuda.synchronize()
            dt_roc = (time.perf_counter() - t0) / 10
            # fp16 baseline (torch matmul, what the fp8 model runs today)
            xf, wf = x.float().half(), w.float().half()
            ws16 = ws.to(torch.float32).reshape(1, -1)
            for _ in range(3): y16 = xf @ wf.T
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): y16 = xf @ wf.T
            torch.cuda.synchronize()
            dt16 = (time.perf_counter() - t0) / 10
            # triton (may fail to build its launcher in this env)
            tri_line = ""
            try:
                y_tri = f_tri(x, w, ws, bias)
                torch.cuda.synchronize()
                same = torch.equal(y_roc, y_tri)
                maxdiff = (y_roc.float() - y_tri.float()).abs().max().item() if not same else 0.0
                for _ in range(3): f_tri(x, w, ws, bias)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10): f_tri(x, w, ws, bias)
                torch.cuda.synchronize()
                dt_tri = (time.perf_counter() - t0) / 10
                tri_line = (f"| triton {dt_tri*1000:7.2f} ms (x{dt_tri/dt_roc:4.1f}) "
                            f"| bit-equal {same}" + ("" if same else f" (maxdiff {maxdiff:.3e})"))
            except Exception as e:
                tri_line = f"| triton FAILED: {type(e).__name__}"
            print(f"{name:<28} rocblas {dt_roc*1000:7.2f} ms | fp16 {dt16*1000:7.2f} ms "
                  f"(x{dt16/dt_roc:4.2f}) {tri_line}")

    if tfk is not None:
        compare_and_time("per-channel 2048x2048x2048", 2048, 2048, 2048)
        compare_and_time("per-channel 4096x2048x8192", 4096, 2048, 8192)
        compare_and_time("per-row     4096x2048x8192", 4096, 2048, 8192, per_row=True)
        compare_and_time("per-row     2048x4096x8192", 2048, 4096, 8192, per_row=True)
