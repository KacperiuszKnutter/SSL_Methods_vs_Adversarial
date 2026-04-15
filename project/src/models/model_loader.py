from project.src.core.experiments_config import ExperimentConfig


# class for:
# 1 ) fetching certain models from solo-learn
# 2 ) potentially load a checkpoint if needed
# 3 ) --||-- set up random weights if needed
# 4) enabling/disabling the projector for training

class ModelLoader:
    def load_model(self, config: ExperimentConfig):
        pass

    def load_checkpoint(self, model, checkpoint_path):
        pass

    def set_projector_mode(self, model, use_projector):
        pass