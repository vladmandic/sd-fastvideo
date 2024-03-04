#!/usr/bin/env python

from types import SimpleNamespace
import sys
import time
import argparse
import nvml
from logger import log
from options import options


# global state vars
args = None
times = SimpleNamespace(decode = 0, encode = 0, load = 0, process = 0, read = 0, save = 0)
queue = SimpleNamespace(decode = None, encode = None, process = None, result = None)
frames = SimpleNamespace(encode = 0, process = 0, decode = 0, save = 0)
processes = SimpleNamespace(encode = None, engine = [], decode = None)


def parse_args():
    global args # pylint: disable=global-statement
    log.info(__file__)
    parser = argparse.ArgumentParser(description = 'sd-fastvideo')
    parser.add_argument('--input', type=str, required=True, help="input video file")
    parser.add_argument('--output', type=str, required=False, help="output folder")
    parser.add_argument('--model', type=str, required=False, help="model file")
    parser.add_argument('--prompt', type=str, required=False, help="prompt")
    parser.add_argument('--pipe', type=int, default=1, help="number of processing pipelines")
    parser.add_argument('--skip', type=int, default=0, help="skip every n frames")
    parser.add_argument('--steps', type=int, default=5, help="scheduler steps")
    parser.add_argument('--batch', type=int, default=1, help="batch size")
    parser.add_argument('--scale', type=float, default=1.0, help="rescale factor")
    parser.add_argument('--strength', type=float, default=0.5, help="denoise strength")
    parser.add_argument('--cfg', type=float, default=6.0, help="classifier free guidance")
    parser.add_argument('--rescale', type=float, default=1.0, help="vae rescale")
    parser.add_argument('--vae', type=str, required=False, help="vae file")
    parser.add_argument("--taesd", action="store_true", help="use taesd vae")
    parser.add_argument("--debug", action="store_true", help="debug logging")
    parser.add_argument("--stablefast", action="store_true", help="use stablefast")
    parser.add_argument("--deepcache", action="store_true", help="use deepcache")
    parser.add_argument("--inductor", action="store_true", help="use torch inductor")
    parser.add_argument('--sampler', type=str, required=False, help="sampler", choices=['lcm', 'deis', 'euler', 'dpm'])
    args = parser.parse_args()
    # set options
    options.level = 'DEBUG' if args.debug else 'INFO'
    options.model = args.model or options.model
    options.sampler = args.sampler or options.sampler
    options._prompt = args.prompt or options._prompt # pylint: disable=protected-access
    options.pipelines = args.pipe
    options.vae = args.vae
    options.taesd = args.taesd
    options.batch = args.batch
    options.strength = args.strength
    options.scale = args.scale
    options.cfg = args.cfg
    options.steps = args.steps
    options.skip = args.skip
    options.rescale = args.rescale
    options.input = args.input
    options.output = args.output
    options.stablefast = args.stablefast
    options.deepcache = args.deepcache
    options.inductor = args.inductor


def get_stats():
    g = nvml.get()[0]
    stats = {
        'frames': { 
            'encode': frames.encode.value,
            'proces': frames.process.value,
            'decode': frames.decode.value,
            'result': frames.save,
        },
        'queue': {
            'encode': queue.encode.qsize(),
            'process': queue.process.qsize(),
            'decode': queue.decode.qsize(),
            'result': queue.result.qsize(),
        },
        'time': {
            'load': f'{times.load.value:.3f}',
            'read': f'{times.read:.3f}',
            'encode': f'{times.encode.value:.3f}',
            'proces': f'{times.process.value:.3f}',
            'decode': f'{times.decode.value:.3f}',
            'save': f'{times.save:.3f}',
        },
        'gpu': {
            'memory': g['memory']['used'],
            'load': g['load']['gpu'],
            'state': g['state'],
        },
    }
    log.info(stats)


def setup_context(context):
    import torch.multiprocessing as mp # pylint: disable=redefined-outer-name
    m = context.Manager()
    times.decode = m.Value('i', 0)
    times.encode = m.Value('i', 0)
    times.load = m.Value('i', 0)
    times.process = m.Value('i', 0)
    times.result = m.Value('i', 0)
    frames.decode = m.Value('i', 0)
    frames.encode = m.Value('i', 0)
    frames.process = m.Value('i', 0)
    queue.process = mp.Queue()
    queue.decode = mp.Queue()
    queue.encode = mp.Queue()
    queue.result = mp.Queue()


def terminate():
    if processes.encode is not None and processes.encode.is_alive():
        log.info('terminate: encode')
        processes.encode.terminate()
    if processes.decode is not None and processes.decode.is_alive():
        log.info('terminate: decode')
        processes.decode.terminate()
    for n, p in enumerate(processes.engine):
        if p is not None and p.is_alive():
            log.info(f'terminate: process={n+1}')
            p.terminate()

def sigint_handler(_signum, _frame): # required for process cleanup
    log.info('sigint')
    terminate()
    time.sleep(0.1)
    sys.exit(0)


def init():
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    mp.freeze_support()
    import env
    env.set_environment()
    log.info('environment setup complete')
    import diffusers
    log.info(f'packages: torch={torch.__version__} diffusers={diffusers.__version__} mp={mp.get_sharing_strategy()}')
    log.setLevel(options.level)
    for gpu in nvml.get():
        log.info(f'gpu: {gpu}')

    # shared context and queues
    ctx = mp.get_context('spawn')
    setup_context(ctx)
    log.info(f'options: {options.get()}')

    # start processes: encode/process/decode
    import pipeline
    if options.vae:
        import vae
        log.info('vae: multiprocess')
        processes.encode = mp.Process(target=vae.encoder, args=(queue.encode, queue.process, times.encode, frames.encode, options), daemon=True) # 1 - encode frames to latents
        processes.decode = mp.Process(target=vae.decoder, args=(queue.decode, queue.result, times.decode, frames.decode, options), daemon=True) # 3 - decode latents to frames
        processes.encode.start()
        processes.decode.start()
        for _i in range(options.pipelines):
            pipe = mp.Process(target=pipeline.process, args=(queue.process, queue.decode, times.process, times.load, frames.process, options), daemon=True) # 0 - combo pipeline
            pipe.start()
            processes.engine.append(pipe)
    elif options.taesd:
        import taesd
        log.info('vae: multiprocess taesd')
        processes.encode = mp.Process(target=taesd.encoder, args=(queue.encode, queue.process, times.encode, frames.encode, options), daemon=True) # 1 - encode frames to latents
        processes.decode = mp.Process(target=taesd.decoder, args=(queue.decode, queue.result, times.decode, frames.decode, options), daemon=True) # 3 - decode latents to frames
        processes.encode.start()
        processes.decode.start()
        for _i in range(options.pipelines):
            pipe = mp.Process(target=pipeline.process, args=(queue.process, queue.decode, times.process, times.load, frames.process, options), daemon=True) # 0 - combo pipeline
            pipe.start()
            processes.engine.append(pipe)
    else:
        log.info('vae: monolithic')
        for _i in range(options.pipelines):
            pipe = mp.Process(target=pipeline.process, args=(queue.process, queue.result, times.process, times.load, frames.process, options), daemon=True) # 0 - combo pipeline
            pipe.start()
            processes.engine.append(pipe)


    import signal
    signal.signal(signal.SIGINT, sigint_handler)

    # wait for model load
    while times.load.value == 0:
        time.sleep(0.1)
    log.info('ready...')


if __name__ == "__main__":
    # init
    parse_args()
    init()

    # validate input
    import media
    video, width, height, num_frames = media.get_video(args.input)
    if video is None or width ==0 or height == 0 or num_frames == 0:
        sys.exit(1)

    # start read/save threads
    get_stats()
    t_start = time.time()
    from threading import Thread
    Thread(target=media.read_frames, args=(video, queue, times), daemon=True).start()
    Thread(target=media.save_frames, args=(queue, times, frames), daemon=True).start()
    time.sleep(1)

    # start monitoring
    while queue.process.qsize() > 0 or queue.decode.qsize() > 0 or queue.encode.qsize() > 0 or queue.result.qsize() > 0:
        time.sleep(0.5)
        get_stats()
    t_end = time.time()

    terminate()
    log.info(f'done: time={t_end-t_start:.3f} frames={frames.save} fps={frames.save/(t_end-t_start):.3f} its={options.steps * frames.save/(t_end-t_start):.3f}')
    sys.exit(0)
