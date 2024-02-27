import time
from logger import log


pipe = None
process_calls = 0


def load(options):
    global pipe # pylint: disable=global-statement
    import torch
    import diffusers
    from hijack import hijack_accelerate
    hijack_accelerate()
    t0 = time.time()
    log.info(f'loading: model="{options.model}" options={options.load_config}') # pylint: disable=protected-access
    pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(options.model, **options.load_config)
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
    log.info(f'model: file="{options.model}" class={pipe.__class__.__name__} device={options.device} time={t1-t0:.3f}') # pylint: disable=protected-access
    log.info(f'sampler: class={pipe.scheduler.__class__.__name__} config={pipe.scheduler.config}')
    if options.fuse:
        pipe.fuse_qkv_projections()
    import compilers
    pipe = compilers.stablefast(pipe, options)
    pipe = compilers.deepcache(pipe, options)
    pipe = compilers.inductor(pipe, options)
    import numpy as np
    _res = pipe(
        prompt=options.batch * ["dummy prompt"],
        image=np.zeros((options.batch, 512, 512, 3), dtype=np.uint8),
    )
    t2 = time.time()
    log.info(f'warmup: model="{options.model}" time={t2-t1:.3f} batch={options.batch}') # pylint: disable=protected-access


def process(in_queue, out_queue, elapsed, load_time, frames, options):
    global process_calls # pylint: disable=global-statement
    import torch
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
                num_images_per_prompt = 1,
                do_classifier_free_guidance = options.cfg > 1,
            )
        if compilers.deepcache_worker is not None:
            compilers.deepcache_worker.enable()
        res = None
        sd_dict = {
            'prompt_embeds': options.prompt_embeds,
            'negative_embeds': options.negative_embeds,
            'image': latents,
            'strength': options.strength,
            'num_inference_steps': options.steps,
            'guidance_scale': options.cfg,
            'generator': torch.Generator(options.device).manual_seed(options.seed),
        }
        sd_dict['output_type'] = 'pil' if options.vae is None and not options.taesd else 'latent'
        try:
            res = pipe(**sd_dict)
            process_calls += 1
            log.debug(f'process i={process_calls} in={len(latents)} out={len(res.images)} time={time.time() - t0}')
        except Exception as e:
            log.error(f'process: {e}')
        del latents
        if compilers.deepcache_worker is not None:
            compilers.deepcache_worker.disable()
        if res is not None and res.images is not None:
            frames.value += len(res.images)
            for image in res.images:
                if isinstance(image, torch.Tensor):
                    image = image.detach().clone()
                out_queue.put((image))
        elapsed.value += time.time() - t0
