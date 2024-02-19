import base64
import torch
import torch.nn as nn
import numpy as np
import cv2
from options import options
from logger import log


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
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)


def load(what: str):
    global vae # pylint: disable=global-statement
    log.info(f'vae load: type={what} device={options.device} dtype={options.dtype}')
    if what == 'encoder':
        vae = TAESD(encoder_path='assets/taesd_encoder.pth', decoder_path=None)
        vae.encoder.to(options.device, options.dtype)
    elif what == 'decoder':
        vae = TAESD(encoder_path=None, decoder_path="assets/taesd_decoder.pth")
        vae.decoder.to(options.device, options.dtype)
    return vae


@torch.no_grad()
def decode(latents):
    if vae is None:
        load('decoder')
    try:
        decoded = 255 * vae.decoder(latents)
        tensors = torch.split(decoded, 1, dim=0)
        images = [t.squeeze(0).permute(1, 2, 0).detach().cpu() for t in tensors]
        images = [t.numpy().astype(np.uint8) for t in images]
    except Exception as e:
        log.error(f'decode: latents={latents} {e}')
        images = []
    return images

@torch.no_grad()
def encode(images):
    if vae is None:
        load('encoder')
    tensors = []
    for data in images:
        b64 = data.decode('utf-8')
        b64 = base64.b64decode(b64.split(",")[1]) # Remove the "data:image/jpeg;base64," prefix
        image = np.frombuffer(b64, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1) # h,w,c=>c,h,w
        tensors.append(tensor)
    batch = torch.stack(tensors).to(options.device, options.dtype) / 255.0
    latents = vae.encoder(batch)
    return latents
