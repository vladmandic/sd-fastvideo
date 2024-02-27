import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from logger import log


encoder_calls = 0
decoder_calls = 0
vae = None


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )


def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESD(nn.Module): # pylint: disable=abstract-method
    latent_magnitude = 2
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth"):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    @staticmethod
    def scale_latents(x): # """raw latents -> [0, 1]"""
        return x.div(TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x): # """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(TAESD.latent_magnitude)


def load(what: str, options):
    global vae # pylint: disable=global-statement
    log.info(f'vae load: type={what} device={options.device} dtype={options.dtype}')
    if what == 'encoder':
        vae = TAESD(encoder_path='assets/taesd_encoder.pth', decoder_path=None)
        vae.encoder.to(options.device, options.dtype)
    elif what == 'decoder':
        vae = TAESD(encoder_path=None, decoder_path="assets/taesd_decoder.pth")
        vae.decoder.to(options.device, options.dtype)
    return vae


def decoder(in_queue, out_queue, elapsed, frames, options): # batch decodes processed images placed in queue
    global decoder_calls # pylint: disable=global-statement
    log.setLevel(options.level)
    import env
    env.set_environment()
    load('decoder', options)
    batch = []
    while True:
        batch.append(in_queue.get())
        if len(batch) == options.batch:
            t0 = time.time()
            latents = torch.stack(batch).to(options.device, options.dtype)
            # if abs(latents.amax()) + abs(latents.amin()) <= 0:
            #    continue # empty image
            with torch.no_grad():
                decoded = vae.decoder(latents)
            decoded = 255.0 * decoded
            tensors = torch.split(decoded, 1, dim=0)
            images = [t.squeeze(0).permute(1, 2, 0).detach().cpu() for t in tensors]
            images = [t.to(torch.float32).numpy().astype(np.uint8) for t in images] # np doesnt support bfloat16
            decoder_calls += 1
            frames.value += len(images)
            for image in images:
                out_queue.put((image))
            elapsed.value += time.time() - t0
            log.debug(f'decode  i={decoder_calls} in={len(latents)} out={len(images)} input={latents.amin():.3f}/{latents.amax():.3f} output={decoded.amin():.3f}/{decoded.amax():.3f} time={time.time() - t0}')
            batch.clear()


def encoder(in_queue, out_queue, elapsed, frames, options): # batch encodes images for processing placed in queue
    global encoder_calls # pylint: disable=global-statement
    log.setLevel(options.level)
    import env
    env.set_environment()
    load('encoder', options)
    images = []
    while True:
        images.append(in_queue.get())
        if len(images) == options.batch:
            t0 = time.time()
            tensors = [TF.to_tensor(i) for i in images]
            batch = torch.stack(tensors).to(options.device, options.dtype)
            with torch.no_grad():
                latents = vae.encoder(batch)
            latents = latents.detach().clone()
            encoder_calls += 1
            frames.value += len(images)
            out_queue.put((latents))
            elapsed.value += time.time() - t0
            log.debug(f'encode  i={encoder_calls} in={len(images)} output={latents.amin():.3f}/{latents.amax():.3f} out={len(latents)} time={time.time() - t0}')
            images.clear()
