import os
import sys
# Chạy từ thư mục root: python -m test.lab4_test
from src.representations.word_embedder import WordEmbedder

def test_vector_for_king(embedder):
    king_vec = embedder.get_vector('king')
    print("Vector for 'king':\n", king_vec)

def test_similarity(embedder):
    sim_kq = embedder.get_similarity('king', 'queen')
    sim_km = embedder.get_similarity('king', 'man')
    print("Similarity(king, queen):", sim_kq)
    print("Similarity(king, man):", sim_km)

def test_most_similar(embedder):
    similar_computer = embedder.get_most_similar('computer', top_n=10)
    print("Top-10 similar to 'computer':")
    if similar_computer is not None:
        for w, s in similar_computer:
            print(f"  {w}: {s:.4f}")
    else:
        print("  'computer' not in vocabulary.")

def test_document_embedding(embedder):
    sentence = "The queen rules the country."
    doc_vec = embedder.embed_document(sentence)
    print("Document embedding for:", sentence)
    print(doc_vec)

def test_word_embedder():
    embedder = WordEmbedder()
    test_vector_for_king(embedder)
    test_similarity(embedder)
    test_most_similar(embedder)
    test_document_embedding(embedder)

if __name__ == "__main__":
    test_word_embedder()