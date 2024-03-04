# Fast Video and WebCam processing using Stable Diffusion

## Highlights

- CLI usage and Web client

### Optimizations

- Torch multiprocessing:
  - separate VAE encode and decode processes (if TAESD is enabled)
  - separate pipeline process (configurable number)
- Python multithreading:
  - separate frame read and frame write threads
- Queues that link read->encode->process->decode->write

### Processing

- Pre-computed prompt embeds (computed on prompt change, not on each generate)
- Optional LCM scheduler (should use SD15 LCM model)
- Optional StableFast
- Optional DeepCache
- Optional Torch compile

### Results

Without too many optimizations (more to come) its already running at ~35 FPS using nVidia RTX4090 at 360x640 (1/2 scaled down video HD resolution)  

> done: time=14.316 frames=508 fps=35.485 its=177.427

And with higher step count and simpler scheduler, overall FPS goes down, but damn look at its!

> done: time=37.310 frames=500 fps=13.401 its=268.024

## CLI usage

> python engine/main.py --help

```log
options:
  -h, --help            show this help message and exit
  --input INPUT         input video file
  --output OUTPUT       output folder
  --model MODEL         model file
  --prompt PROMPT       prompt
  --pipe PIPE           number of processing pipelines
  --skip SKIP           skip every n frames
  --steps STEPS         scheduler steps
  --batch BATCH         batch size
  --scale SCALE         rescale factor
  --strength STRENGTH   denoise strength
  --cfg CFG             classifier free guidance
  --vae                 use full vae
  --debug               debug logging
  --stablefast          use stablefast
  --deepcache           use deepcache
  --inductor            use torch inductor
  --sampler {lcm,deis,euler,dpm}
  ```

*Note*: For any options not exposed by cli args, modify `engine/options.py` as necessary  
For example, compile modes, torch dtype, etc.

### Example

> python engine/main.py --scale 0.5 --skip 0 --batch 8 --pipe 1 --input TheShimmy.mp4 --output /tmp/frames

```log
12:14:25-750151 INFO     environment setup log
12:14:25-841154 INFO     packages: torch=2.2.0+cu121 diffusers=0.27.0.dev0 mp=log
12:14:25-962239 INFO     gpu: {'name': 'NVIDIA GeForce RTX 4090', 'version': {'cuda': 12040, 'driver': '551.52', 'vbios': '95.02.3c.40.b8', 'rom': 'G002.0000.00.03', 'capabilities': (8, 9)}, 'pci': {'link': 4, 'width': 16, 'busid': '00000000:01:00.0', 'deviceid': 646189278}, 'memory': {'total': 24564.0, 'free': 22768.81, 'used': 1795.19}, 'clock': {'gpu': [210, 3375], 'sm': [210, 3375], 'memory': [405, 10501]}, 'load': {'gpu': log
                         'memory': 14, 'temp': 50, 'fan': 31}, 'power': [29.34, 405.0], 'state': 'gpu idlelog
12:14:25-990180 INFO     input video: path=TheShimmy.mp4 frames=509 fps=60 size=720x1280 codec=log
12:14:26-603977 INFO     options: {'level': 'INFO', 'model': 'assets/photonLCM_v10.safetensors', 'prompt': 'sexy girl dancing', 'negative': '', 'width': 360.0, 'height': 640.0, 'steps': 5, 'strength': 0.2, 'cfg': 6.0, 'batch': 8, 'device': 'cuda', 'dtype': 'torch.float16', 'channels_last': False, 'inductor': False, 'stablefast': False, 'deepcache': False, 'fuse': log
12:14:26-605654 INFO     vae: taesd log
12:14:27-269564 INFO     vae load: type=encoder device=cuda dtype=torch.log
12:14:27-285394 INFO     vae load: type=decoder device=cuda dtype=torch.log
12:14:27-841512 INFO     loading: model="assets/photonLCM_v10.safetensors" options={'low_cpu_mem_usage': True, 'torch_dtype': torch.float16, 'safety_checker': None, 'requires_safety_checker': False, 'load_safety_checker': False, 'load_connected_pipeline': True, 'use_safetensors': True, 'extract_ema': True, 'config_files': {'v1': 'configs/v1-inference.yaml', 'v2': 'configs/v2-inference-768-v.yaml', 'xl': 'configs/sd_xl_base.yamllog
                         'xl_refiner': 'configs/sd_xl_refiner.yamllog
12:14:29-945010 INFO     model: file="assets/photonLCM_v10.safetensors" class=StableDiffusionImg2ImgPipeline device=cuda time=2.log
12:14:29-946444 INFO     sampler: class=LCMScheduler config=FrozenDict([('num_train_timesteps', 1000), ('beta_start', 0.00085), ('beta_end', 0.012), ('beta_schedule', 'scaled_linear'), ('trained_betas', None), ('original_inference_steps', 50), ('clip_sample', False), ('clip_sample_range', 1.0), ('set_alpha_to_one', False), ('steps_offset', 1), ('prediction_type', 'epsilon'), ('thresholding', False), ('dynamic_thresholding_ratio', 0.995log
                         ('sample_max_value', 1.0), ('timestep_spacing', 'leading'), ('timestep_scaling', 10.0), ('rescale_betas_zero_snr', Falselog
12:14:39-359392 INFO     warmup: model="assets/photonLCM_v10.safetensors" time=9.414 batch=log
12:14:39-449658 INFO     readylog
12:14:39-466074 INFO     thread start: log
12:14:39-466833 INFO     thread start: log
12:14:39-467305 INFO     save: fn=/tmp/frames/TheShimmy00000.log
12:14:39-905348 INFO     thread done: read time=0.log
...
12:14:46-621771 INFO     {'frames': {'encode': 504, 'proces': 160, 'decode': 160, 'result': 160}, 'queue': {'encode': 0, 'process': 42, 'decode': 0, 'result': 0}, 'time': {'load': '12.095', 'read': '0.418', 'encode': '3.602', 'proces': '6.371', 'decode': '2.342', 'save': '0.003'}, 'gpu': {'memory': 9442.89, 'load': 98, 'state': 'sw power caplog
...
12:14:53-920597 INFO     terminate: log
12:14:53-921165 INFO     terminate: log
12:14:53-921675 INFO     terminate: process=log
12:14:53-922218 INFO     done: time=14.455 frames=488 fps=33.761 its=168.804
```

## TODO

- Web client
  - All communication between browser and backend is done using raw websockets in real-time
  - Client maintains constant frame rate based on forward-adjusted server latency
- IPAdapter
- ControlNet
- Video output
- LoRA
