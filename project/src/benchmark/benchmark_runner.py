from project.src.core.experiments_config import ExperimentConfig

# checking the model in the registry, load it, run it on validation set, generate embeddings, liczy benchmark
class BenchmarkRunner:
    def __init__(self, model_registry, model_loader, dataset_manager, extractor, evaluator):
        pass

    def run(self, config: ExperimentConfig) -> dict:
        # Returns benchmark results and paths to generated artifacts.
        pass