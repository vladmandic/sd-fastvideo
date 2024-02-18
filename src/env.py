import os
import torch
from logger import log


def set_environment():
    log.debug('env setup')
    os.environ.setdefault('ACCELERATE', 'True')
    os.environ.setdefault('ATTN_PRECISION', 'fp16')
    os.environ.setdefault('CUDA_AUTO_BOOST', '1')
    os.environ.setdefault('CUDA_CACHE_DISABLE', '0')
    os.environ.setdefault('CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT', '0')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')
    os.environ.setdefault('FORCE_CUDA', '1')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('K_DIFFUSION_USE_COMPILE', '0')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
    os.environ.setdefault('PYTHONHTTPSVERIFY', '0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')
    os.environ.setdefault('SAFETENSORS_FAST_GPU', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('USE_TORCH', '1')
    os.environ.setdefault('UVICORN_TIMEOUT_KEEP_ALIVE', '60')
    os.environ.setdefault('KINETO_LOG_LEVEL', '3')
    os.environ.setdefault('DO_NOT_TRACK', '1')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
