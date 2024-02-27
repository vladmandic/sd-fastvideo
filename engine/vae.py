import time
import torch
from logger import log


encoder_calls = 0
decoder_calls = 0
vae = None
proc = None


def load(options):
    global vae, proc # pylint: disable=global-statement
    log.info(f'vae load: model={options.vae} device={options.device} dtype={options.dtype}')
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    vae = AutoencoderKL.from_single_file(options.vae, **options.load_config)
    vae.to(options.device, options.dtype)
    proc = VaeImageProcessor()
    return vae


def decoder(in_queue, out_queue, elapsed, frames, options): # batch decodes processed images placed in queue
    global decoder_calls # pylint: disable=global-statement
    log.setLevel(options.level)
    import env
    env.set_environment()
    load(options)
    batch = []
    while True:
        latent = in_queue.get()
        t0 = time.time()
        latents = latent.unsqueeze(0).to(options.device, options.dtype) / vae.config.scaling_factor
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]
        decoded = (decoded + 0.5) / 2
        image = 255 * decoded.clamp(0, 1).squeeze(0)
        image = image.detach().cpu().permute(1, 2, 0).float().numpy().astype('uint8')
        batch.append(in_queue.get())
        decoder_calls += 1
        frames.value += 1
        out_queue.put((image))
        elapsed.value += time.time() - t0
        log.debug(f'decode  i={decoder_calls} input={latent.amin():.3f}/{latent.amax():.3f} output={decoded.amin():.3f}/{decoded.amax():.3f} time={time.time() - t0}')


def encoder(in_queue, out_queue, elapsed, frames, options): # batch encodes images for processing placed in queue
    global encoder_calls # pylint: disable=global-statement
    log.setLevel(options.level)
    import env
    env.set_environment()
    load(options)
    while True:
        image = in_queue.get()
        t0 = time.time()
        # tensor = proc.preprocess(image / 255).to(options.device, options.dtype)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        tensor = (tensor / 255).to(options.device, options.dtype)
        with torch.no_grad():
            latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
        encoder_calls += 1
        frames.value += 1
        log.debug(f'encode  i={encoder_calls} input={tensor.amin()}/{tensor.amax()} output={latent.amin():.3f}/{latent.amax():.3f} time={time.time() - t0}')
        out_queue.put((latent.detach().clone()))
        elapsed.value += time.time() - t0
