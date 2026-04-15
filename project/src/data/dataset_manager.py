from project.src.core.experiments_config import ExperimentConfig

# returns DataLoaders for pretrain, test/benchmark and downstream eval

class DatasetManager:
    def get_train_loader(self, config: ExperimentConfig):
        pass

    def get_eval_loader(self, config: ExperimentConfig):
        pass

    def get_embedding_loader(self, config: ExperimentConfig):
        pass