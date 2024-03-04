import torch


class Options():
    def __init__(self):
        # general
        self.level = 'DEBUG'
        self.scale = 0.5

        # media
        self.input = None
        self.output = None
        self.skip = 0

        # generate defaults
        self.model = 'assets/photonLCM_v10.safetensors'
        self.sampler = 'lcm'
        self._prompt = 'color sketch of a beautiful, sexy young girl with blonde hair performing yoga, naked, nsfw, perfect face, rich colors, cinestil, outdoors'
        self._negative = 'clothes, shirt, pants, ugly, deformed, mutilated'
        self._seed = -1
        self.width: int = 512 # unused
        self.height: int = 512 # unused
        self.steps: int = 5
        self.strength: float = 0.5
        self.cfg: float = 4.0
        self.rescale: float = 1.0

        # optimizations
        self.vae = None
        self.taesd = False
        self.device: str = 'cuda'
        self.dtype = torch.float16
        self.pipelines: int = 1 # number of processing instances to start
        self.batch: int = 1 # number of items to hold per buffer and process in parallel
        self.channels_last = True # use cuddn channels last
        self.inductor = False # compile model using torch inductor
        self.stablefast = False # compile model using stablefast
        self.deepcache = False # enable deepcache optimizations
        self.fuse = True # enable torch kvq fuse optimization

        # internal
        self.prompt_embeds = None
        self.negative_embeds = None
        self.noise = None
        self.vae_scaling_factor = 0.18215
        self.load_config = {
            "low_cpu_mem_usage": True,
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            'extract_ema': True,
            'original_config_file': 'configs/v1-inference.yaml',
        }
        self.scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'trained_betas': None,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'set_alpha_to_one': False,
            'steps_offset': 1,
            'prediction_type': 'epsilon',
            'thresholding': False,
            'dynamic_thresholding_ratio': 0.995,
            'sample_max_value': 1.0,
            'timestep_spacing': 'leading',
            'rescale_betas_zero_snr': False,
            # 'timestep_scaling': 10.0,
            # 'original_inference_steps': 50,
        }

    def get(self):
        return {
            "level": self.level,
            "model": self.model,
            "prompt": self.prompt,
            "negative": self.negative,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "strength": self.strength,
            "cfg": self.cfg,
            "rescale": self.rescale,
            "batch": self.batch,
            "device": self.device,
            "dtype": str(self.dtype),
            "channels_last": self.channels_last,
            "inductor": self.inductor,
            "stablefast": self.stablefast,
            "deepcache": self.deepcache,
            "fuse": self.fuse,
        }

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        self.prompt_embeds = None
        self.negative_embeds = None

    @property
    def negative(self):
        return self._negative

    @negative.setter
    def negative(self, value):
        self._negative = value
        self.prompt_embeds = None
        self.negative_embeds = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.generator = torch.Generator(self.device).manual_seed(self.seed)


options = Options()
