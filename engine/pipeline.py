import time
import random
from logger import log, console


pipe = None
process_calls = 0


def load(options):
    global pipe # pylint: disable=global-statement
    import torch
    import diffusers
    from hijack import hijack_accelerate
    from pipelines.sd import StableDiffusionImg2ImgPipeline
    hijack_accelerate()
    t0 = time.time()
    log.info(f'loading: model="{options.model}" options={options.load_config}') # pylint: disable=protected-access
    # pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(options.model, **options.load_config)
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(options.model, **options.load_config)

    pipe.set_progress_bar_config(disable=True)
    if options.sampler == 'euler':
        cls = diffusers.EulerAncestralDiscreteScheduler
    elif options.sampler == 'lcm':
        cls = diffusers.LCMScheduler
    elif options.sampler == 'deis':
        cls = diffusers.DEISMultistepScheduler
    elif options.sampler == 'dpm':
        cls = diffusers.DPMSolverMultistepScheduler
    pipe.scheduler = cls.from_config(options.scheduler_config)
    if options.channels_last:
        pipe.to(memory_format=torch.channels_last)
    pipe.to(options.device, options.dtype)
    t1 = time.time()
    if options.seed == -1:
        options.seed = int(random.randrange(4294967294))
    log.info(f'model: file="{options.model}" class={pipe.__class__.__name__} device={options.device} time={t1-t0:.3f}') # pylint: disable=protected-access
    log.info(f'sampler: class={pipe.scheduler.__class__.__name__} config={pipe.scheduler.config} seed={options.seed}')
    if options.fuse:
        pipe.fuse_qkv_projections()
    import compilers
    pipe = compilers.stablefast(pipe, options)
    pipe = compilers.deepcache(pipe, options)
    pipe = compilers.inductor(pipe, options)
    """
    import numpy as np
    _res = pipe(
        prompt=options.batch * ["dummy prompt"],
        image=np.zeros((options.batch, 512, 512, 3), dtype=np.uint8),
    )

    t2 = time.time()
    log.info(f'warmup: model="{options.model}" time={t2-t1:.3f} batch={options.batch}') # pylint: disable=protected-access
    """

def process(in_queue, out_queue, elapsed, load_time, frames, options):
    global process_calls # pylint: disable=global-statement
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    log.setLevel(options.level)
    import env
    env.set_environment()
    t0 = time.time()
    import compilers
    load(options)
    load_time.value = time.time() - t0
    while True:
        latents = in_queue.get()
        t0 = time.time()
        if options.prompt_embeds is None or options.prompt_embeds.shape[0] != len(latents):
            options.prompt_embeds, options.negative_embeds = pipe.encode_prompt(
                prompt = len(latents) * [options.prompt],
                negative_prompt = len(latents) * [options.negative],
                device = options.device,
                do_classifier_free_guidance = options.cfg > 1,
            )
        if options.noise is None:
            options.noise = randn_tensor(latents.shape, generator=options.generator, device=torch.device(options.device), dtype=options.dtype)
        if compilers.deepcache_worker is not None:
            compilers.deepcache_worker.enable()
        sd_dict = {
            'prompt_embeds': options.prompt_embeds,
            'negative_prompt_embeds': options.negative_embeds,
            'image': latents,
            'strength': options.strength,
            'num_inference_steps': options.steps,
            'guidance_scale': options.cfg,
            'generator': options.generator,
            'noise': options.noise,
        }
        images = []
        try:
            images = pipe(**sd_dict)
            process_calls += 1
            log.debug(f'process i={process_calls} in={len(latents)} out={len(images)} time={time.time() - t0}')
        except Exception as e:
            log.error(f'process: {e}')
            console.print_exception(show_locals=False, max_frames=10, extra_lines=1, suppress=[], theme="ansi_dark", word_wrap=False, width=console.width)
        del latents
        if compilers.deepcache_worker is not None:
            compilers.deepcache_worker.disable()
        frames.value += len(images)
        for image in images:
            if isinstance(image, torch.Tensor):
                image = image.detach().float().cpu()
            out_queue.put((image))
        elapsed.value += time.time() - t0
