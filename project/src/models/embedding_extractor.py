
# get the model, train on the data set, collect the embeddings # save it

class EmbeddingExtractor:
    def evaluate_model(self, model, dataloader, use_projector):
        pass
    # returns for instance embeddings as nparr, labels as nparr, opt metadata

# maybe let it use two options -> 1) extracting backbone embeddings 2) extracting embeddings after using the projector.
# param: extract_layer="backbone" | "projector"
