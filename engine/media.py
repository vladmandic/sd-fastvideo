import os
import sys
import time
import cv2
from logger import log
from options import options


def get_video(fn: str):
    try:
        stream = cv2.VideoCapture(fn)
        if not stream.isOpened():
            log.error(f'video open failed: path={fn}')
            return None, 0, 0, 0
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(stream.get(cv2.CAP_PROP_FPS))
        w, h = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cc = stream.get(cv2.CAP_PROP_FOURCC)
        cc_bytes = int(cc).to_bytes(4, byteorder=sys.byteorder) # convert code to a bytearray
        codec = cc_bytes.decode() # decode byteaarray to a string
        log.info(f'input video: path={fn} frames={total_frames} fps={fps} size={w}x{h} codec={codec}')
        return stream, w, h, total_frames
    except Exception as e:
        log.error(f'video open failed: path={fn} {e}')
        return None, 0, 0, 0


def read_frames(cv2_video, queue, times):
    from PIL import Image
    log.info('thread start: read')
    status, frame = cv2_video.read() # first frame
    n = 0
    t0 = time.time()
    batch = []
    while status:
        if n % (options.skip + 1) == 0:
            log.debug(f'enqueue: frame={n}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _c = frame.shape
            frame = cv2.resize(frame, (int(8 * options.scale * w // 8), int(8 * options.scale * h // 8)))
            if options.vae is not None or options.taesd:
                queue.encode.put((frame.copy()))
            else:
                image = Image.fromarray(frame)
                batch.append(image)
                if len(batch) == options.batch:
                    queue.process.put(batch.copy())
                    batch.clear()
        n += 1
        status, frame = cv2_video.read()
    times.read = time.time() - t0
    log.info(f'thread done: read time={times.read:.3f}')


def save_frames(queue, times, frames):
    from PIL import Image
    log.info('thread start: save')
    n = 0
    if options.output is not None:
        if not os.path.exists(options.output):
            os.makedirs(options.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(options.input))[0]
        basename = os.path.join(options.output, basename)
        log.info(f'save: fn={basename}{0:05d}.jpg')
    else:
        log.info(f'save: location={options.output}')
    while True:
        frame = queue.result.get()
        if not options.output:
            continue # just empty the queue
        try:
            t0 = time.time()
            n += 1
            fn = f'{basename}{n:05d}.jpg'
            if isinstance(frame, Image.Image):
                frame.save(fn)
            else:
                cv2.imwrite(fn, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            log.debug(f'save: frame={n} fn={fn}')
            times.save += time.time() - t0
            frames.save += 1
        except Exception as e:
            log.error(f'save: frame={frame} {e}')
