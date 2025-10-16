# Lab 4

## 📁 Cấu trúc dự án

```
lab4/
├── data/                           # Dữ liệu training
│   ├── c4-train.00000-of-01024-30K.json.gz
│   └── en_ewt-ud-train.txt
├── src/                            # Source code chính
│   ├── preprocessing/
│   │   └── tokenizer.py           # Tokenizer class
│   └── representations/
│       └── word_embedder.py       # WordEmbedder với pre-trained models
├── test/                           # Scripts demo, test, trực quan hóa
│   ├── lab4_test.py               # Test pre-trained embeddings
│   ├── lab4_embedding_training_demo.py  # Train với Gensim
│   └── lab4_spark_word2vec_demo.py      # Train với PySpark
├── results/                        # Models đã train
│   ├── word2vec_ewt.model         # Gensim Word2Vec model
│   └── spark_word2vec_model/      # Model Spark Word2Vec
├── scripts/                        # Setup scripts (bat, sh, ps1)
│   ├── setup_glove.*              # Script tải/cài GloVe tùy môi trường
│   └── cleanup_glove.*            # Script dọn dẹp GloVe tùy môi trường
├── requirements.txt                # Dependencies
└── README.md                       # Tài liệu dự án
```

## 🔧 Chi tiết các bước thực hiện

### Bước 1: Chuẩn bị dữ liệu và môi trường

1. **Cài đặt dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   - Cài đặt các thư viện cần thiết: gensim, pyspark, scikit-learn, matplotlib

2. **Setup GloVe pre-trained model:**

   ```bash
   # Linux/Mac
   bash scripts/setup_glove.sh

   # Windows
   scripts/setup_glove.bat
   ```

### Bước 2: Preprocessing dữ liệu

1. **Tokenization:**

   - File: `src/preprocessing/tokenizer.py`
   - Chức năng: Tách từ, loại bỏ stopwords, chuẩn hóa text

2. **Load Word Embeddings:**

   - File: `src/representations/word_embedder.py`
   - Chức năng: Nhúng từ sang vector

### Bước 3: Training Word2Vec Models

#### 3.1 Training với Gensim

**Script:** `test/lab4_embedding_training_demo.py`

**Các tham số quan trọng:**

- `vector_size=100`: Kích thước vector embedding
- `window=5`: Kích thước context window
- `min_count=1`: Tần suất tối thiểu của từ
- `workers=4`: Số thread xử lý song song
- `sg=0`: CBOW (0) hoặc Skip-gram (1)

**Quy trình:**

1. Load và preprocess dữ liệu
2. Khởi tạo và train model Word2Vec
3. Lưu model vào `results/word2vec_ewt.model`
4. Test similarity giữa các từ

#### 3.2 Training với PySpark

**Script:** `test/lab4_spark_word2vec_demo.py`

**Quy trình:**

1. Khởi tạo SparkSession
2. Tạo DataFrame từ dữ liệu đã tokenize
3. Sử dụng `Word2Vec` của Spark MLlib
4. Training và lưu model vào `results/spark_word2vec_model/`
5. Đánh giá performance

### Bước 4: Sử dụng Pre-trained Embeddings

**Script:** `test/lab4_test.py`

**Chức năng:**

1. **Load GloVe vectors:**

   ```python
   from src.representations.word_embedder import WordEmbedder

   embedder = WordEmbedder()
   embedder.load_glove('path/to/glove-wiki-gigaword-50')
   ```

2. **Tính similarity:**

   ```python
   sim = embedder.similarity('king', 'queen')
   print(f"Similarity: {sim:.4f}")
   ```

3. **Tìm từ tương tự:**
   ```python
   similar = embedder.find_similar_words('king', top_k=5)
   ```

### Bước 5: Trực quan hóa Embeddings

**Techniques sử dụng:**

1. **PCA Dimensionality Reduction:**

   - Giảm từ 100D xuống 2D để visualization
   - Giữ lại thông tin quan trọng nhất

2. **Scatter Plot:**

   - Hiển thị các từ trong không gian 2D
   - Quan sát clustering và relationships

**Lưu ý Spark trên Windows:**

- Cần cài winutils.exe và thiết lập HADOOP_HOME
- Nếu gặp lỗi native Hadoop, xem: https://github.com/steveloughran/winutils

## 🎯 Đánh giá và So sánh Models

### 1. Metrics đánh giá

- **Cosine Similarity:** Đo độ tương đồng giữa các word vectors
- **Analogical Reasoning:** Test khả năng "king - man + woman = queen"
- **Word Similarity Tasks:** Benchmark trên các dataset chuẩn

### 2. So sánh các approaches

| Method                | Pros                         | Cons                                  | Performance  | Use Case              |
| --------------------- | ---------------------------- | ------------------------------------- | ------------ | --------------------- |
| **GloVe Pre-trained** | Chất lượng cao, ready-to-use | Kích thước lớn, fixed vocabulary      | **Tốt nhất** | Production, research  |
| **Spark Word2Vec**    | Scalable, distributed        | Setup phức tạp, overhead              | Trung bình   | Big data scenarios    |
| **Gensim Word2Vec**   | Flexible, customizable       | Cần dữ liệu train, thời gian training | **Kém nhất** | Domain-specific tasks |

**Lý do hiệu suất:**

- **GloVe Pre-trained:** Được train trên corpus khổng lồ (6B tokens), có chất lượng semantic representation tốt nhất
- **Gensim Word2Vec:** Train trên dataset nhỏ (en_ewt-ud-train.txt), vocabulary hạn chế, chất lượng phụ thuộc vào dữ liệu training

## 📝 Cách chạy từng script

### 1. Test Pre-trained Embeddings

```bash
python test/lab4_test.py
```

**Mục đích:** Test GloVe embeddings, tính similarity scores

### 2. Training Demo với Gensim

```bash
python test/lab4_embedding_training_demo.py
```

**Mục đích:** Train Word2Vec model từ scratch, save model

### 3. Training với Spark

```bash
python test/lab4_spark_word2vec_demo.py
```

**Mục đích:** Distributed training với Spark MLlib
