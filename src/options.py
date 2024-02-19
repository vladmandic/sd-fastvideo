from types import SimpleNamespace
import torch


class Options():
    def __init__(self):
        # general
        self.log_delay: int = 10

        # generate defaults
        self._model = 'assets/photonLCM_v10.safetensors'
        self._prompt = 'watercolor painting of old man looking at camera'
        self._negative = ''
        self._seed = 424242
        self.width: int = 512
        self.height: int = 512
        self.steps: int = 3
        self.strength: float = 0.5
        self.cfg: float = 0.0

        # optimizations
        self.device: str = 'cuda'
        self.dtype = torch.float16
        self.buffers: int = 2 # number of round-robin buffers
        self.batch: int = 1 # number of items to hold per buffer and process in parallel
        self.channels_last = False # use cuddn channels last
        self.inductor = False # compile model using torch inductor
        self.stablefast = False # compile model using stablefast
        self.deepcache = False # enable deepcache optimizations
        self.fuse = True # enable torch kvq fuse optimization

        # internal
        self.prompt_embeds = None
        self.negative_embeds = None
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
            "model": self.model,
            "prompt": self.prompt,
            "negative": self.negative,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "strength": self.strength,
            "cfg": self.cfg,

            "buffers": self.buffers,
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
stats_dict = {
    'load': 0,
    'warmup': 0,
    'encode': 0,
    'decode': 0,
    'generate': 0,
    'prompt': 0,
    'network': 0,
    'frames': 0,
}
stats = SimpleNamespace(**stats_dict)
