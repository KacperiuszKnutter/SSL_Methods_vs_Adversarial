
# Dictionary that holds list of supported models

class ModelRegistry:
    SUPPORTED_MODELS = {
        "simclr": "solo.methods.simclr",
        "byol": "solo.methods.byol",
        "barlow_twins": "solo.methods.barlow_twins",
        "vicreg": "solo.methods.vicreg",
        "dino": "solo.methods.dino",
        "mae": "solo.methods.mae",
    }

    @classmethod
    def is_method_supported(cls, model_name : str) -> bool:
        return model_name in cls.SUPPORTED_MODELS