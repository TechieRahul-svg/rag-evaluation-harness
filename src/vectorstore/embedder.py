from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            MODEL_NAME
        )


    def embed_documents(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=True
        )


    def embed_query(self, query):

        return self.model.encode(
            query
        )