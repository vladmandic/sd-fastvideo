import time
from threading import Thread
# from multiprocessing import Process
# from queue import Queue
from torch.multiprocessing import Process, Queue
import numpy as np
from options import options, stats
from logger import log


loaded = ''
pipe = None
buffers = options.buffers * [Queue()]
decode_queue = Queue()
encode_queue = Queue()
upload_queue = Queue()


def gb(val: float):
    return round(val / 1024 / 1024 / 1024, 2)


def queue(data: bytes): # add uploaded image to either queue
    if pipe is None:
        return False
    for b in buffers:
        if b.qsize() < options.batch:
            b.put((data))
            return True
    return False


def decoder(in_queue, out_queue, elapsed): # decodes processed images
    from taesd import decode
    while True:
        latents = in_queue.get()
        t0 = time.time()
        images = decode(latents)
        del latents
        for image in images:
            out_queue.put((image))
        elapsed.value += time.time() - t0
        # decode_queue.task_done()


def encoder(in_buffers, out_queue, elapsed): # encodes images for processing
    from taesd import encode
    while True:
        images = []
        for b in in_buffers:
            if b.qsize() == options.batch:
                while b.qsize() > 0:
                    images.append(b.get())
        if len(images) > 0:
            t0 = time.time()
            latents = encode(images)
            while len(images) > 0:
                del images[0]
            out_queue.put((latents))
            elapsed.value += time.time() - t0
        else:
            time.sleep(0.05)


def load(model: str):
    import torch
    import diffusers
    from hijack import hijack_accelerate
    hijack_accelerate()
    t0 = time.time()
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
    import compilers
    pipe = compilers.stablefast(pipe)
    pipe = compilers.deepcache(pipe)
    pipe = compilers.inductor(pipe)
    _res = pipe(
        prompt=options.batch * ["dummy prompt"],
        image=np.zeros((options.batch, 512, 512, 3), dtype=np.uint8),
    )
    t2 = time.time()
    stats.load += t1 - t0
    stats.warmup += t2 - t1
    log.info(f'warmup: model="{model}" time={t2-t1:.3f} batch={options.batch}') # pylint: disable=protected-access
    loaded = model
    options._model = model # pylint: disable=protected-access


def process():
    import torch
    import compilers
    while True:
        if pipe is None:
            time.sleep(1)
            continue
        images = encode_queue.get()
        t0 = time.time()
        if options.prompt_embeds is None or options.prompt_embeds.shape[0] != images.shape[0]:
            options.prompt_embeds, options.negative_embeds = pipe.encode_prompt(
                prompt = images.shape[0] * [options.prompt],
                negative_prompt = len(images) * [options.negative],
                device = options.device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = options.cfg > 1,
            )
            stats.prompt += time.time() - t0
        t1 = time.time()
        if compilers.deepcache_worker is not None:
            compilers.deepcache_worker.enable()
        try:
            res = pipe(
                # prompt=len(images) * [options.prompt],
                # negative=len(images) * [options.negative],
                prompt_embeds = options.prompt_embeds,
                negative_embeds = options.negative_embeds,
                image = images,
                strength = options.strength,
                num_inference_steps=options.steps,
                guidance_scale = options.cfg,
                generator = options.generator,
                output_type='latent',
            )
            del images
            # torch.cuda.synchronize(options.device)
            if compilers.deepcache_worker is not None:
                compilers.deepcache_worker.disable()
            if res is not None and res.images is not None:
                t2 = time.time()
                stats.generate += t2 - t1
                decode_queue.put((res.images))
                t3 = time.time()
                stats.frames += len(res.images)
                its = options.strength * options.steps * len(res.images) / (t3-t0)
                gpu = torch.cuda.mem_get_info()
                log.info(f'process: time={t3-t0:.3f} steps={options.steps} images={len(res.images)} its={its:.3f} used={gb(gpu[1] - gpu[0])} total={gb(gpu[1])}')
            else:
                log.error(f'process: res={res}')
        except Exception as e:
            log.error(f'process: {e}')
        # encode_queue.task_done()


def start(decode_elapsed, encode_elapsed):
    # from multiprocessing import Manager
    process_thread = Thread(target=process, daemon=True)
    process_thread.start()
    log.info('Started thread: process')
    decode_process = Process(target=decoder, args=(decode_queue, upload_queue, decode_elapsed), daemon=True)
    decode_process.start()
    log.info('Started worker: decoder')
    encode_process = Process(target=encoder, args=(buffers, encode_queue, encode_elapsed), daemon=True)
    encode_process.start()
    log.info('Started worker: encoder')
