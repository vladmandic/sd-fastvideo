import torch

class Options():
    def __init__(self):
        self._model = 'assets/photonLCM_v10.safetensors'
        self._prompt = 'watercolor painting of old man looking at camera'
        self._negative = ''
        self._seed = 424242
        self.taesd = True
        self.device = 'cuda'
        self.batch: int = 8
        self.width: int = 640
        self.height: int = 480
        self.steps: int = 5
        self.strength: float = 0.5
        self.cfg: float = 0.0
        self.log_delay: int = 10
        self.prompt_embeds = None
        self.negative_embeds = None
        self.channels_last = False
        self.inductor = False
        self.stablefast = False
        self.deepcache = False
        self.fuse = True
        self.dtype = torch.float16
        self.generator = torch.Generator(self.device).manual_seed(self.seed)
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

    def get(self):
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "batch": self.batch,
            "model": self.model,
            "prompt": self.prompt,
            "negative": self.negative,
            "strength": self.strength,
            "steps": self.steps,
            "cfg": self.cfg,
            "deepcache": self.deepcache,
            "stablefast": self.stablefast,
            "inductor": self.inductor,
        }

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        import engine
        engine.load(value)

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        self.prompt_embeds = None

    @property
    def negative(self):
        return self._negative

    @negative.setter
    def negative(self, value):
        self._negative = value
        self.negative_embeds = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

options = Options()
