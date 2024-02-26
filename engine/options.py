import torch


class Options():
    def __init__(self):
        # general
        self.level = 'DEBUG'
        self.scale = 0.5

        # generate defaults
        self.model = 'assets/photonLCM_v10.safetensors'
        self._prompt = 'sexy girl dancing'
        self._negative = ''
        self._seed = 424242
        self.width: int = 512
        self.height: int = 512
        self.steps: int = 5
        self.strength: float = 0.2
        self.cfg: float = 4.0

        # optimizations
        self.vae = False
        self.device: str = 'cuda'
        self.dtype = torch.float16
        self.pipelines: int = 1 # number of processing instances to start
        self.batch: int = 1 # number of items to hold per buffer and process in parallel
        self.channels_last = False # use cuddn channels last
        self.inductor = False # compile model using torch inductor
        self.stablefast = False # compile model using stablefast
        self.deepcache = False # enable deepcache optimizations
        self.fuse = False # enable torch kvq fuse optimization

        # internal
        self.prompt_embeds = None
        self.negative_embeds = None
        self.load_config = {
            "low_cpu_mem_usage": True,
            "torch_dtype": self.dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
            "load_safety_checker": False,
            "load_connected_pipeline": True,
            "use_safetensors": True,
            'extract_ema': True,
            'config_files': {
                'v1': 'configs/v1-inference.yaml',
                'v2': 'configs/v2-inference-768-v.yaml',
                'xl': 'configs/sd_xl_base.yaml',
                'xl_refiner': 'configs/sd_xl_refiner.yaml',
            }
        }
        self.scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'trained_betas': None,
            'original_inference_steps': 50,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'set_alpha_to_one': False,
            'steps_offset': 1,
            'prediction_type': 'epsilon',
            'thresholding': False,
            'dynamic_thresholding_ratio': 0.995,
            'sample_max_value': 1.0,
            'timestep_spacing': 'leading',
            'timestep_scaling': 10.0,
            'rescale_betas_zero_snr': False,
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
