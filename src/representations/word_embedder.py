import gensim.downloader as api
import os
import numpy as np

from src.preprocessing.tokenizer import Tokenizer

class WordEmbedder:
    def __init__(self, model_name: str='glove-wiki-gigaword-50'):
        """
        Initialize the WordEmbedder with a pre-trained word embedding model.

        Args:
            model_name (str): The name of the pre-trained model to load. Default is 'glove-wiki-gigaword-50'.
        """
        if model_name not in api.info()['models']:
            raise ValueError(f"Model '{model_name}' not found in gensim's available models.")
        if self.is_model_cached(model_name):
            print(f"Loading model '{model_name}'...")
            self.model = api.load(model_name)
        else:
            print(f"Model not found. Downloading...")
            self.model = api.load(model_name)
            print(f"Model '{model_name}' downloaded and loaded successfully!")

    @staticmethod
    def is_model_cached(model_name: str) -> bool:
        """
        Check if a model is already cached locally.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is cached, False otherwise.
        """
        try:
            cache_dir = api.base_dir
            if not os.path.exists(cache_dir):
                return False
            model_path = os.path.join(cache_dir, model_name)
            if os.path.exists(model_path) and os.path.isdir(model_path):
                model_files = os.listdir(model_path)
                return len(model_files) > 0
            return False
        except Exception:
            return False

    def get_vector(self, word: str) -> np.ndarray:
        """
        Returns the embedding vector for a given word.

        Args:
            word (str): The word to get the vector for.

        Returns:
            numpy.ndarray: The embedding vector as numpy array.
        """
        if word in self.model:
            return self.model[word]
        else:
            return None
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Compute the cosine similarity between two words.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: Cosine similarity between the two words. Returns None if either word is not in the vocabulary.
        """
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        else:
            return None

    def get_most_similar(self, word: str, top_n: int=10) -> list[tuple[str, float]]:
        """
        Get the top-N most similar words to the given word.

        Args:
            word (str): The word to find similar words for.
            top_n (int): The number of top similar words to return. Default is 10.

        Returns:
            list of tuples: List of (word, similarity) tuples for the most similar words.
        """
        if word in self.model:
            return self.model.most_similar(positive=[word], topn=top_n)
        else:
            return None

    def embed_document(self, document: str) -> np.ndarray:
        """
        Embed a document by averaging the embeddings of its words.

        Args:
            document (str): The input document as a string.

        Returns:
            numpy.ndarray: The averaged embedding vector for the document.
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(document)
        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
            else:
                vectors.append(np.zeros(self.model.vector_size))
        return np.mean(vectors, axis=0)

__all__ = ['WordEmbedder']