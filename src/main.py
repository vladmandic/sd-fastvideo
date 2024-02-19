#!/usr/bin/env python

import time
import torch
import torch.multiprocessing as mp
import diffusers
from logger import log
from options import options, stats


if __name__ == "__main__":
    t0 = time.time()
    log.info(__file__)
    mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')
    m = ctx.Manager()
    decode_elapsed = m.Value('i', 0)
    encode_elapsed = m.Value('i', 0)
    from env import set_environment
    set_environment()
    log.info(f'packages: torch={torch.__version__} diffusers={diffusers.__version__}')
    log.info(f'options: {options.get()}')
    import server
    master, app, ws = server.httpd()
    import network
    network.mount(app)
    import nvml
    for gpu in nvml.get():
        log.info(f'gpu: {gpu}')
    mp.freeze_support()
    import engine
    engine.load(options.model)
    engine.start(decode_elapsed, encode_elapsed)
    log.info('ready...')
    while True:
        try:
            alive = master.thread.is_alive()
            requests = master.server_state.total_requests
        except Exception:
            alive = False
            requests = 0
        if round(time.time()) % options.log_delay == 0:
            stats.decode = decode_elapsed.value
            stats.encode = encode_elapsed.value
            log.debug(f'server: alive={alive} requests={requests} uptime={round(time.time() - t0)} clients={len(ws.active)} stats={stats.__dict__}')
        if not alive:
            log.info('exiting...')
            break
        time.sleep(1)
