import time
import logging
from logger import log
from options import options


deepcache_worker = None


def deepcache(pipe):
    global deepcache_worker # pylint: disable=global-statement
    if not options.deepcache:
        return pipe
    try:
        from DeepCache import DeepCacheSDHelper
    except Exception as e:
        log.warning(f"compile: task=deepcache {e}")
        return pipe
    if deepcache_worker is not None:
        deepcache_worker.disable()
    deepcache_worker = DeepCacheSDHelper(pipe=pipe)
    deepcache_worker.set_params(cache_interval=3, cache_branch_id=0)
    log.info(f"compile: task=deepcache config={deepcache_worker.params}")
    return pipe


def stablefast(pipe):
    import torch
    if not options.stablefast:
        return pipe
    try:
        import sfast.compilers.stable_diffusion_pipeline_compiler as sf
    except Exception as e:
        log.warning(f"compile: task=stablefast {e}")
        return pipe
    config = sf.CompilationConfig.Default()
    try:
        import xformers # pylint: disable=unused-import # noqa: F401
        config.enable_xformers = True
    except Exception:
        pass
    try:
        import triton # pylint: disable=unused-import # noqa: F401
        config.enable_triton = True
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    config.enable_cuda_graph = True # shared.opts.cuda_compile_fullgraph
    config.enable_jit_freeze = True # shared.opts.diffusers_eval
    config.memory_format = torch.channels_last if options.channels_last else torch.contiguous_format
    try:
        t0 = time.time()
        pipe = sf.compile(pipe, config)
        t1 = time.time()
        log.info(f"compile: task=stablefast config={config.__dict__} time={t1-t0:.2f}")
    except Exception as e:
        log.error(f"compile: task=stablefast error: {e}")
    return pipe


def inductor(pipe):
    import torch
    if not options.inductor:
        return pipe
    try:
        t0 = time.time()
        import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
        torch._dynamo.reset() # pylint: disable=protected-access
        log.debug(f"compile available backends: {torch._dynamo.list_backends()}") # pylint: disable=protected-access

        def torch_compile_model(model):
            return torch.compile(model, mode='max-autotune', backend='inductor', dynamic=False, fullgraph=True)

        log_level = logging.CRITICAL # pylint: disable=protected-access
        if hasattr(torch, '_logging'):
            torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
        torch._dynamo.config.verbose = False # pylint: disable=protected-access
        torch._dynamo.config.suppress_errors = True # pylint: disable=protected-access
        try:
            torch._inductor.config.conv_1x1_as_mm = True # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_tuning = True # pylint: disable=protected-access
            torch._inductor.config.epilogue_fusion = False # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_check_all_directions = True # pylint: disable=protected-access
            torch._inductor.config.use_mixed_mm = True # pylint: disable=protected-access
            # torch._inductor.config.force_fuse_int_mm_with_mul = True # pylint: disable=protected-access
        except Exception as e:
            log.error(f"inductor config error: {e}")
        pipe.unet = torch_compile_model(pipe.unet)
        pipe.text_encoder = torch_compile_model(pipe.text_encoder)
        t1 = time.time()
        log.info(f"compile: task=inductor config={torch._inductor.config._config} time={t1-t0:.2f}") # pylint: disable=protected-access
    except Exception as e:
        log.warning(f"compile error: {e}")
    return pipe
