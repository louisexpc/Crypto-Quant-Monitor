# import os, torch

# print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
# print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

# n = torch.cuda.device_count()
# print("visible GPUs:", n)
# for i in range(n):
#     name = torch.cuda.get_device_name(i)
#     cap  = torch.cuda.get_device_capability(i)
#     print(f"cuda:{i} -> {name}, sm_{cap[0]}{cap[1]}")

# import torch
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("GPU:", torch.cuda.get_device_name(0))
# print("Compute capability:", torch.cuda.get_device_capability(0))

def setup_cuda_acceleration():
    import torch
    import warnings

    # 開啟 TF32 加速（Ampere+ GPU 有效）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"  # 或 "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # # 關閉 TorchInductor 的 Triton 編譯，防止 crash
    # try:
    #     import torch._inductor
    #     torch._inductor.config.triton = False
    # except Exception as e:
    #     warnings.warn(f"Could not disable triton: {e}")

    # SDP 設定（可跳過，視模型而定）
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass