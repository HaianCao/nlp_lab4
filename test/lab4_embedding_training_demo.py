import os
import sys
from pathlib import Path
from gensim.models import Word2Vec
import re
from functools import reduce
from typing import Iterator, List, Tuple, Dict, Any

def read_paragraphs(file_path: str) -> list:
    """
    Read paragraphs from a text file. Paragraphs are separated by blank lines.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list[str]: List of paragraph strings.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs

def split_into_sentences(text: str) -> Iterator[str]:
    """
    Split a paragraph into sentences.

    Args:
        text (str): The paragraph text.

    Yields:
        str: Each sentence.
    """
    sentences = re.split(r'[.!?]+["\']?\s*', text)
    for sent in sentences:
        sent = sent.strip()
        if sent:
            yield sent

def tokenize_sentences(sentences: Iterator[str]) -> Iterator[List[str]]:
    """
    Tokenize sentences into words.

    Args:
        sentences (Iterator[str]): An iterator over sentences.

    Yields:
        List[str]: List of words in each sentence.
    """
    for sent in sentences:
        words = re.findall(r'\b\w+\b', sent.lower())
        if len(words) > 1:
            yield words


def load_training_data(file_path: str) -> List[List[str]]:
    """
    Load and preprocess training data from a text file (paragraph-based, blank lines separate paragraphs).

    Args:
        file_path (str): Path to the text file.

    Returns:
        List[List[str]]: List of tokenized sentences.
    """
    paragraphs = read_paragraphs(file_path)
    all_tokenized = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        all_tokenized.extend(tokenize_sentences(sentences))
    return all_tokenized

def calculate_statistics(sentences: List[List[str]]) -> Dict[str, int]:
    """
    Calculate dataset statistics.

    Args:
        sentences (List[List[str]]): List of tokenized sentences.

    Returns:
        Dict[str, int]: Dictionary with statistics: number of sentences, total words, unique words
    """
    total_sentences = len(sentences)
    total_words = sum(len(sent) for sent in sentences)
    unique_words = len(set(word for sent in sentences for word in sent))
    
    return {
        'sentences': total_sentences,
        'total_words': total_words,
        'unique_words': unique_words
    }


def print_statistics(stats: Dict[str, int]) -> None:
    """
    Print dataset statistics.

    Args:
        stats (Dict[str, int]): Dictionary with statistics.
    """
    print(f"📊 {stats['sentences']} sentences, "
          f"{stats['total_words']} words, "
          f"{stats['unique_words']} unique")

def create_model_config() -> Dict[str, Any]:
    """
    Create Word2Vec model configuration.

    Returns:
        Dict[str, Any]: Configuration parameters for Word2Vec model.
    """
    return {
        'vector_size': 100,
        'window': 5,
        'min_count': 2,
        'workers': 4,
        'epochs': 10,
        'sg': 0  # CBOW
    }


def train_model(sentences: List[List[str]], config: Dict[str, Any]) -> Word2Vec:
    """
    Train Word2Vec model with given configuration.

    Args:
        sentences (List[List[str]]): List of tokenized sentences.
        config (Dict[str, Any]): Configuration parameters for Word2Vec model.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    print(f"\n🤖 Training model...")
    model = Word2Vec(sentences=sentences, **config)
    print(f"✅ Vocabulary size: {len(model.wv)}")
    return model


def save_model(model: Word2Vec, output_path: str) -> Word2Vec:
    """
    Save trained model to disk.

    Args:
        model (Word2Vec): The trained Word2Vec model.
        output_path (str): Path to save the model.

    Returns:
        Word2Vec: The saved Word2Vec model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"💾 Saved: {os.path.basename(output_path)}")
    return model


def load_model(model_path: str) -> Word2Vec:
    """
    Load a saved Word2Vec model from disk.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Word2Vec: The loaded Word2Vec model.
    """
    print(f"\n📂 Loading model from: {os.path.basename(model_path)}")
    model = Word2Vec.load(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Vocabulary size: {len(model.wv)}")
    print(f"   Vector dimension: {model.wv.vector_size}")
    return model

def demonstrate_usage(model: Word2Vec) -> None:
    """
    Demonstrate how to use the trained model to find similar words and solve analogies.
    
    Args:
        model (Word2Vec): The trained Word2Vec model.
    """
    test_words = ['king', 'woman', 'good', 'man', 'queen']
    print("\n📍 Finding Similar Words:\n")
    for word in test_words:
        if word in model.wv:
            print(f"  '{word}' → Similar words:")
            similar = model.wv.most_similar(word, topn=5)
            for similar_word, score in similar:
                print(f"    • {similar_word}: {score:.4f}")
            print()
        else:
            print(f"  '{word}' → ❌ Not in vocabulary\n")
    
    print("🧩 Solving Analogies:\n")
    analogies = [
        ('king', 'man', 'queen', 'woman'),  # king - man + woman ≈ queen
        ('good', 'better', 'bad', 'worse'),  # good - better + bad ≈ worse
        ('man', 'men', 'woman', 'women'),    # man - men + woman ≈ women
    ]
    for word_a, word_b, word_c, expected in analogies:
        print(f"  {word_a} - {word_b} + {word_c} ≈ ?")
        if all(w in model.wv for w in [word_a, word_b, word_c]):
            try:
                result = model.wv.most_similar(
                    positive=[word_a, word_c],
                    negative=[word_b],
                    topn=3
                )
                print(f"    Top predictions:")
                for pred_word, score in result:
                    marker = "✅" if pred_word == expected else "  "
                    print(f"    {marker} {pred_word}: {score:.4f}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            missing = [w for w in [word_a, word_b, word_c] if w not in model.wv]
            print(f"    ❌ Missing words: {', '.join(missing)}")
        print()
    print("="*60)


def training_pipeline(data_path: str, output_path: str) -> Word2Vec:
    """
    Complete training pipeline:
    load data -> calculate stats -> train model -> save model

    Args:
        data_path (str): Path to the training data file.
        output_path (str): Path to save the trained model.

    Returns:
        Word2Vec: The trained Word2Vec model.
    """
    print("🎓 Word2Vec Training Pipeline\n")
    print(f"📁 Data: {Path(data_path).name}")
    print(f"💾 Output: {Path(output_path).name}\n")
    
    sentences = load_training_data(data_path)
    
    stats = calculate_statistics(sentences)
    print_statistics(stats)
    
    config = create_model_config()
    model = train_model(sentences, config)
    
    model = save_model(model, output_path)
    
    return model

def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "en_ewt-ud-train.txt"
    output_path = project_root / "results" / "word2vec_ewt.model"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    model = training_pipeline(str(data_path), str(output_path))
    demonstrate_usage(model)
    
    loaded_model = load_model(str(output_path))
    
    print("\n🧪 Testing loaded model:")
    test_word = 'good'
    if test_word in loaded_model.wv:
        similar = loaded_model.wv.most_similar(test_word, topn=3)
        print(f"   Similar to '{test_word}':")
        for word, score in similar:
            print(f"     • {word}: {score:.4f}")
    
    print("="*60)
    print(f"\n✅ Complete!")


if __name__ == "__main__":
    main()