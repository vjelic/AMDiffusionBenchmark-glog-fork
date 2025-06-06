import src.models.flux
import src.models.hunyuan
import src.models.mochi_1
import src.models.stablediffusionxl
import src.models.wan2_1


class ModelManager:
    def __init__(self):
        """Initialize model manager with the available models mapping."""
        self.model_mapping = {
            "flux-dev": {
                "class": src.models.flux.FluxModel,
                "output": "image",
                "input": "text",
            },
            "hunyuan-video": {
                "class": src.models.hunyuan.HunyuanVideoModel,
                "output": "video",
                "input": "text",
            },
            "stable-diffusion-xl": {
                "class": src.models.stablediffusionxl.StableDiffusionXLModel,
                "output": "image",
                "input": "text",
            },
            "mochi-1": {
                "class": src.models.mochi_1.MochiModel,
                "output": "video",
                "input": "text",
            },
            "wan2_1-i2v": {
                "class": src.models.wan2_1.Wan2_1_I2V_Model,
                "output": "video",
                "input": "image",
            },
        }

    def get_model(self, model_name):
        """Get the model class and type based on the model name."""
        if model_name not in self.model_mapping:
            raise NotImplementedError(f"`{model_name}` is not supported")

        model_info = self.model_mapping[model_name]
        return model_info["class"], model_info["input"], model_info["output"]

    def get_available_models(self):
        """Get the list of available model names."""
        return list(self.model_mapping.keys())
