#!/usr/bin/env python

import time
import torch
import diffusers
from logger import log
from server import httpd
from options import options
import engine
import nvml
from env import set_environment


if __name__ == "__main__":
    t0 = time.time()
    log.info(__file__)
    set_environment()
    log.info(f'packages: torch={torch.__version__} diffusers={diffusers.__version__}')
    log.info(f'options: {options.get()}')
    server, app, ws = httpd()
    for gpu in nvml.get():
        log.info(f'gpu: {gpu}')
    engine.load(options.model)
    log.info('ready...')
    while True:
        try:
            alive = server.thread.is_alive()
            requests = server.server_state.total_requests
        except Exception:
            alive = False
            requests = 0
        if round(time.time()) % options.log_delay == 0:
            log.debug(f'server: alive={alive} requests={requests} uptime={round(time.time() - t0)} clients={len(ws.active)} frames={engine.frames} queue={len(engine.b1)}/{len(engine.b2)}')
        if not alive:
            log.info('exiting...')
            break
        time.sleep(1)
