import os
import sys
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lower, regexp_replace, split


def init_spark(app_name: str = "Word2Vec Fast Training"):
    """
    Initialize Spark session optimized for local mode.

    Args:
        app_name (str): Name of the Spark application.
    """
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")  # Use all available cores
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Disable Arrow for stability
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def preprocess_text(df: DataFrame) -> DataFrame:
    """
    Clean and tokenize text data for Word2Vec.

    Args:
        df (DataFrame): Input DataFrame with a 'text' column.

    Returns:
        DataFrame: Processed DataFrame with a 'words' column.
    """
    return (
        df.select("text")
        .filter(col("text").isNotNull())
        .withColumn("text_lower", lower(col("text")))
        .withColumn(
            "text_clean",
            regexp_replace(col("text_lower"), r"[^a-z0-9\s]", " ")
        )
        .withColumn("words", split(col("text_clean"), r"\s+"))
        .select("words")
        .filter(col("words").isNotNull())
    )


def train_word2vec(df_processed: DataFrame, vector_size: int = 100, window: int = 5, min_count: int = 2, max_iter: int = 5) -> Word2Vec:
    """
    Train a Word2Vec model.

    Args:
        df_processed (DataFrame): Preprocessed DataFrame with a 'words' column.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        max_iter (int): Number of iterations (epochs) over the corpus.

    Returns:
        Word2VecModel: Trained Word2Vec model.
    """
    word2vec = Word2Vec(
        vectorSize=vector_size,
        windowSize=window,
        minCount=min_count,
        maxIter=max_iter,
        inputCol="words",
        outputCol="word_vectors"
    )
    model = word2vec.fit(df_processed)
    return model


def load_json_gz_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load JSON gzip data file using Spark's native reader.
    
    Args:
        spark (SparkSession): Active Spark session.
        file_path (str): Path to the JSON gzip file.
    
    Returns:
        DataFrame: DataFrame with 'text' column.
    """
    df = spark.read.option("compression", "gzip").json(file_path)
    if 'text' in df.columns:
        return df.select("text")
    else:
        print(f"Available columns: {df.columns}")
        return df

def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "c4-train.00000-of-01024-30K.json.gz"
    output_path = project_root / "results" / "spark_word2vec_model"

    print("=" * 70)
    print("🚀 Word2Vec Training with PySpark")
    print("=" * 70)

    print("\n📦 Initializing Spark...")
    spark = init_spark()
    print("✅ Spark initialized\n")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        spark.stop()
        return

    print(f"📁 Loading dataset...")
    df = load_json_gz_data(spark, str(data_path))
    row_count = df.count()
    print(f"✅ Loaded {row_count:,} documents\n")

    print("🧹 Preprocessing text...")
    df_processed = preprocess_text(df).cache()
    df_processed.count()  # trigger cache
    print("✅ Preprocessing done and cached\n")

    print("🤖 Training Word2Vec model...")
    model = train_word2vec(df_processed, vector_size=100, window=5, max_iter=5)
    vocab_size = model.getVectors().count()
    print(f"✅ Model trained. Vocabulary size: {vocab_size:,}\n")

    print("💾 Saving model...")
    if output_path.exists():
        shutil.rmtree(output_path)
    model.save(str(output_path))
    print(f"✅ Model saved to: {output_path.name}\n")

    print("🎯 Demo: Top synonyms for 'computer' (if exists)")
    try:
        synonyms = model.findSynonyms("computer", 5)
        rows = synonyms.collect()
        if rows:
            for r in rows:
                print(f"   • {r['word']}: {r['similarity']:.4f}")
        else:
            print("⚠️ Word not found in vocabulary.")
    except Exception as e:
        print(f"⚠️ Cannot find synonyms: {e}")
        model.getVectors().show(5, truncate=False)

    spark.stop()
    print("\n🛑 Spark stopped. Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
