import csv
from pathlib import Path

import numpy as np


def build_vocabulary(corpus):
    # Tách từ trong corpus và chỉ giữ lại mỗi từ một lần.
    words = set()
    for sentence in corpus:
        words.update(sentence.lower().split())

    # Bỏ "the" vì vocabulary đã chọn trong W3 không gồm từ này.
    words.discard("the")

    # Sắp xếp để thứ tự hàng/cột ổn định giữa các lần chạy.
    vocabulary = sorted(words)
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    return vocabulary, word_to_index


def build_cooccurrence_matrix(corpus, vocabulary, window):
    # Tạo ma trận: hàng là target word, cột là context word.
    if window < 1:
        raise ValueError("window phải lớn hơn hoặc bằng 1")

    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)

    for sentence in corpus:
        tokens = sentence.lower().split()

        for target_position, target_word in enumerate(tokens):
            if target_word not in word_to_index:
                continue

            target_index = word_to_index[target_word]
            start = max(0, target_position - window)
            end = min(len(tokens), target_position + window + 1)

            for context_position in range(start, end):
                # Không tính chính target word là context của nó.
                if context_position == target_position:
                    continue

                context_word = tokens[context_position]
                if context_word in word_to_index:
                    context_index = word_to_index[context_word]
                    matrix[target_index, context_index] += 1

    return matrix


def cosine_similarity(x, y):
    # Tính cosine similarity và xử lý vector có độ dài bằng 0.
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("Hai vector phải có cùng số chiều")

    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return float(dot_product / (norm_x * norm_y))


def most_similar(word, matrix, vocabulary, top_k=5):
    # Tìm các từ có cosine similarity cao nhất với từ đầu vào.
    if word not in vocabulary or top_k <= 0:
        return []

    word_index = vocabulary.index(word)
    target_vector = matrix[word_index]
    similarities = []

    for other_index, other_word in enumerate(vocabulary):
        if other_word == word:
            continue

        score = cosine_similarity(target_vector, matrix[other_index])
        similarities.append((other_word, score))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]


def print_matrix(matrix, vocabulary):
    # In ma trận cùng nhãn hàng và cột.
    column_width = max(8, max(len(word) for word in vocabulary) + 2)
    header = " " * column_width
    header += "".join(f"{word:>{column_width}}" for word in vocabulary)
    print(header)

    for word, row in zip(vocabulary, matrix):
        values = "".join(f"{value:>{column_width}}" for value in row)
        print(f"{word:>{column_width}}{values}")


def main():
    # Corpus nhỏ được dùng trong Experiment 1.
    corpus = [
        "the cat eats fish",
        "the cat likes milk",
        "the dog eats meat",
        "the dog likes fish",
    ]

    # Tạo vocabulary và mapping word sang vị trí trong vector.
    vocabulary, word_to_index = build_vocabulary(corpus)
    windows = [1, 2, 5]
    results = []

    print("Vocabulary:", vocabulary)
    print("Word to index:", word_to_index)

    for window in windows:
        matrix = build_cooccurrence_matrix(corpus, vocabulary, window)
        non_zero = np.count_nonzero(matrix)

        cat_vector = matrix[word_to_index["cat"]]
        dog_vector = matrix[word_to_index["dog"]]
        eats_vector = matrix[word_to_index["eats"]]
        likes_vector = matrix[word_to_index["likes"]]

        sim_cat_dog = cosine_similarity(cat_vector, dog_vector)
        sim_eats_likes = cosine_similarity(eats_vector, likes_vector)

        print(f"\n===== window = {window} =====")
        print("Vocabulary size:", len(vocabulary))
        print("Matrix shape:", matrix.shape)
        print("Number of non-zero values:", non_zero)
        print("Co-occurrence matrix:")
        print_matrix(matrix, vocabulary)

        print("Vector cat:", cat_vector)
        print("Vector dog:", dog_vector)
        print("Vector eats:", eats_vector)
        print("Vector likes:", likes_vector)
        print(f"cosine_similarity(cat, dog): {sim_cat_dog:.6f}")
        print(f"cosine_similarity(eats, likes): {sim_eats_likes:.6f}")
        print("Most similar to cat:", most_similar("cat", matrix, vocabulary, top_k=5))

        results.append(
            {
                "window": window,
                "vocabulary_size": len(vocabulary),
                "matrix_rows": matrix.shape[0],
                "matrix_cols": matrix.shape[1],
                "non_zero": int(non_zero),
                "sim_cat_dog": sim_cat_dog,
                "sim_eats_likes": sim_eats_likes,
            }
        )

    # Ghi bảng so sánh ba window ra results.csv cạnh file này.
    results_path = Path(__file__).with_name("results.csv")
    columns = [
        "window",
        "vocabulary_size",
        "matrix_rows",
        "matrix_cols",
        "non_zero",
        "sim_cat_dog",
        "sim_eats_likes",
    ]

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nĐã lưu kết quả vào: {results_path}")


if __name__ == "__main__":
    main()
