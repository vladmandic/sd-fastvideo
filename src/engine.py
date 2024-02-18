import time
import threading
from queue import Queue
from PIL import Image
import torch
import diffusers
from options import options
from logger import log
from hijack import hijack_accelerate
import compilers

b1 = []
b2 = []
pipe: diffusers.StableDiffusionPipeline = None

loaded = ''
frames = 0
hijack_accelerate()
decode_queue = Queue()


def gb(val: float):
    return round(val / 1024 / 1024 / 1024, 2)


def queue(image: Image):
    if pipe is None:
        return False
    if len(b1) < options.batch:
        b1.append(image)
        return True
    elif len(b2) < options.batch:
        b2.append(image)
        return True
    return False


def load(model: str):
    global pipe, loaded # pylint: disable=global-statement
    if loaded == model and pipe is not None:
        return
    t0 = time.time()
    log.info(f'loading: model="{model}" options={options.load_config}') # pylint: disable=protected-access
    pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(model, **options.load_config)
    # pipe.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=40, colour='#327fba')
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
    if options.channels_last:
        pipe.to(memory_format=torch.channels_last)
    pipe.to(options.device, options.dtype)
    t1 = time.time()
    log.info(f'loaded: model="{model}" class={pipe.__class__.__name__} sampler={pipe.scheduler.__class__.__name__} device={options.device} time={t1-t0:.3f}') # pylint: disable=protected-access
    if options.fuse:
        pipe.fuse_qkv_projections()
    pipe = compilers.stablefast(pipe)
    pipe = compilers.deepcache(pipe)
    pipe = compilers.inductor(pipe)
    _res = pipe(
        prompt=options.batch * ["dummy prompt"],
        image=options.batch * [Image.new("RGB", (options.width, options.height))],
    )
    t2 = time.time()
    log.info(f'warmup: model="{model}" time={t2-t1:.3f}') # pylint: disable=protected-access
    loaded = model
    options._model = model # pylint: disable=protected-access


def decoder():
    from network import manager
    from taesd import decode
    while True:
        images = decode_queue.get()
        decoded = decode(images)
        for image in decoded:
            manager.queue.put((image))
        decode_queue.task_done()


def process():
    global frames # pylint: disable=global-statement
    while True:
        if pipe is None:
            time.sleep(1)
            continue
        images = []
        if len(b1) == options.batch:
            images = b1
        if len(b2) == options.batch:
            images = b2
        if len(images) == 0:
            time.sleep(1)
            continue
        t0 = time.time()
        if options.prompt_embeds is None or options.negative_embeds is None:
            if compilers.deepcache_worker is not None:
                compilers.deepcache_worker.disable()
                compilers.deepcache_worker.enable()
            options.prompt_embeds, options.negative_embeds = pipe.encode_prompt(
                prompt = len(images) * [options.prompt],
                negative_prompt = len(images) * [options.negative],
                device = options.device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = options.cfg > 1,
            )
        res = pipe(
            # prompt=len(images) * [options.prompt],
            # negative=len(images) * [options.negative],
            prompt_embeds=options.prompt_embeds,
            negative_embeds=options.negative_embeds,
            image=images,
            strength=options.strength,
            num_inference_steps=options.steps,
            guidance_scale=options.cfg,
            generator=options.generator,
            output_type='latent' if options.taesd else 'pil',
        )
        images.clear()
        decode_queue.put((res.images))
        t1 = time.time()
        frames += len(res.images)
        its = options.strength * options.steps * len(res.images) / (t1-t0)
        gpu = torch.cuda.mem_get_info()
        log.info(f'process: time={t1-t0:.3f} images={len(res.images)} its={its:.3f} used={gb(gpu[1] - gpu[0])} total={gb(gpu[1])}')


process_thread = threading.Thread(target=process, daemon=True)
process_thread.start()
decode_thread = threading.Thread(target=decoder, daemon=True)
decode_thread.start()
