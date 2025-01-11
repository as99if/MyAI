from langchain.embeddings import OpenAIEmbeddings

class Vectorizer:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()

    def vectorize(self, text):
        """
        Converts text into an embedding vector.
        
        Args:
            text (str): The text to be vectorized.
        
        Returns:
            list: The embedding vector.
        """
        return self.embedding_model.embed(text)
