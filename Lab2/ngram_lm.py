from collections import Counter
import math


class NGramLanguageModel:
    """Mô hình ngôn ngữ unigram, bigram hoặc trigram dùng MLE."""

    def __init__(self, n, smoothing="mle"):
        # n cho biết mô hình đang dùng một, hai hay ba từ.
        if n not in (1, 2, 3):
            raise ValueError("n chỉ nhận giá trị 1, 2 hoặc 3")
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing chỉ nhận 'mle' hoặc 'laplace'")

        self.n = n
        self.smoothing = smoothing
        self.unk_token = "<unk>"
        self.vocabulary = []
        self.vocabulary_set = set()
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.bigram_context_counts = Counter()
        self.trigram_context_counts = Counter()
        self.counts = Counter()
        self.probabilities = {}
        self.total_tokens = 0

    def _to_tokens(self, text_or_tokens):
        # Chuyển câu hoặc danh sách từ thành các token chữ thường.
        if isinstance(text_or_tokens, str):
            return text_or_tokens.lower().split()
        return [str(word).lower() for word in text_or_tokens]

    def build_vocabulary(self, corpus):
        # Vocabulary là tập các từ duy nhất trong corpus.
        words = set()
        for sentence in corpus:
            words.update(self._to_tokens(sentence))

        self.vocabulary = sorted(words)
        return self.vocabulary

    def count_ngrams(self, tokens):
        # Đếm các chuỗi n từ liên tiếp; tuple giúp giữ đúng thứ tự từ.
        tokens = self._to_tokens(tokens)
        counts = Counter()

        for start in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[start:start + self.n])
            counts[ngram] += 1

        return counts

    def fit(self, corpus):
        # Đếm n-gram trong từng câu riêng để không ghép qua ranh giới câu.
        if isinstance(corpus, str):
            corpus = [corpus]

        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()
        self.counts.clear()
        self.probabilities.clear()

        tokenized_corpus = [self._to_tokens(sentence) for sentence in corpus]
        self.vocabulary = sorted(
            {word for tokens in tokenized_corpus for word in tokens} | {self.unk_token}
        )
        self.vocabulary_set = set(self.vocabulary)

        for tokens in tokenized_corpus:
            self.unigram_counts.update(tokens)

            if self.n >= 2:
                for start in range(len(tokens) - 1):
                    bigram = tuple(tokens[start:start + 2])
                    self.bigram_counts[bigram] += 1
                    self.bigram_context_counts[bigram[0]] += 1

            if self.n >= 3:
                for start in range(len(tokens) - 2):
                    trigram = tuple(tokens[start:start + 3])
                    context = trigram[:2]
                    self.trigram_counts[trigram] += 1
                    self.trigram_context_counts[context] += 1

            # Hàm đếm n-gram được gọi theo n đã chọn cho model.
            self.counts.update(self.count_ngrams(tokens))

        self.total_tokens = sum(self.unigram_counts.values())

        # Lưu xác suất các bậc thấp để tính những từ đầu câu của trigram.
        self.train_unigram()
        self.train_bigram()
        self.train_trigram()
        return self

    def train_unigram(self):
        # P(word) bằng số lần word chia cho tổng số token.
        if self.total_tokens == 0:
            return self.probabilities

        for word, count in self.unigram_counts.items():
            self.probabilities[((), word)] = count / self.total_tokens
        return self.probabilities

    def train_bigram(self):
        # P(word|previous) bằng count bigram chia cho count của previous.
        for (previous, word), count in self.bigram_counts.items():
            denominator = self.bigram_context_counts[previous]
            if denominator:
                self.probabilities[((previous,), word)] = count / denominator
        return self.probabilities

    def train_trigram(self):
        # P(word|hai từ trước) bằng count trigram chia cho count context.
        for (first, second, word), count in self.trigram_counts.items():
            context = (first, second)
            denominator = self.trigram_context_counts[context]
            if denominator:
                self.probabilities[(context, word)] = count / denominator
        return self.probabilities

    def probability(self, context, word):
        # Nếu n-gram chưa gặp trong corpus, MLE trả về zero probability.
        context_tokens = self._to_tokens(context) if context else []
        word = str(word).lower()
        if word not in self.vocabulary_set:
            word = self.unk_token
        context_tokens = [
            token if token in self.vocabulary_set else self.unk_token
            for token in context_tokens
        ]

        # Unigram dùng context rỗng; bigram/trigram lấy phần context cuối.
        if self.n == 1:
            context_key = ()
        else:
            context_key = tuple(context_tokens[-(self.n - 1):])

        # Ở đầu câu chưa đủ context, dùng xác suất bậc thấp hơn.
        if not context_key:
            probability_key = ((), word)
        elif len(context_key) == 1:
            probability_key = (context_key, word)
        else:
            probability_key = (context_key[-2:], word)

        if self.smoothing == "laplace":
            # Laplace smoothing cộng thêm 1 để tránh xác suất bằng 0.
            # Đồng thời điều chỉnh lại toàn bộ phân phối xác suất.
            vocabulary_size = len(self.vocabulary)

            if not context_key:
                count = self.unigram_counts[word]
                context_count = self.total_tokens
            elif len(context_key) == 1:
                count = self.bigram_counts[(context_key[0], word)]
                context_count = self.unigram_counts[context_key[0]]
            else:
                count = self.trigram_counts[(context_key[-2], context_key[-1], word)]
                context_count = self.trigram_context_counts[context_key[-2:]]

            return (count + 1) / (context_count + vocabulary_size)

        return self.probabilities.get(probability_key, 0.0)

    def sentence_probability(self, sentence):
        # Nhân xác suất từng từ để lấy xác suất của toàn bộ câu.
        tokens = self._to_tokens(sentence)
        result = 1.0

        for index, word in enumerate(tokens):
            context = tokens[:index]
            word_probability = self.probability(context, word)
            if word_probability == 0:
                return 0.0
            result *= word_probability

        return result

    def sentence_log_probability(self, sentence):
        # Cộng log xác suất để tránh tích nhiều số nhỏ bị underflow.
        tokens = self._to_tokens(sentence)
        result = 0.0

        for index, word in enumerate(tokens):
            context = tokens[:index]
            word_probability = self.probability(context, word)
            if word_probability == 0:
                # Nếu xác suất bằng 0 thì log(0) không xác định.
                # Trả về -inf để thể hiện câu có xác suất bằng 0.
                return -math.inf
            result += math.log(word_probability)

        return result

    def next_word_distribution(self, context):
        # Trả về các từ đã thấy có thể đứng sau context và xác suất của chúng.
        context_tokens = self._to_tokens(context) if context else []
        distribution = {}

        for word in self.vocabulary:
            word_probability = self.probability(context_tokens, word)
            if word_probability > 0:
                distribution[word] = word_probability

        # Xếp từ có xác suất cao lên trước để dễ đọc kết quả.
        return dict(
            sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        )


if __name__ == "__main__":
    # Ví dụ nhỏ để chạy thử file trực tiếp.
    corpus = [
        "the cat eats fish",
        "the cat likes fish",
        "the dog eats meat",
    ]

    model = NGramLanguageModel(2)
    model.fit(corpus)

    print("P(eats | cat):", model.probability(["cat"], "eats"))
    print("P(the cat eats fish):", model.sentence_probability("the cat eats fish"))
    print("Log P(the cat eats fish):", model.sentence_log_probability("the cat eats fish"))
    print("Next word after 'the':", model.next_word_distribution("the"))
