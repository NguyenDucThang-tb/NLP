# Calculations - tự hoàn thành trước khi chạy code

Đây là file chính theo tên deliverable trong đề. Không lấy đáp án từ code/AI
rồi chép vào; hãy tự tính và ghi cách làm của em.

## Exercise 1 - Count vector

Vocabulary: `[cat, dog, eats, fish, likes]`.

- D1: `cat eats fish` -> [................................................................]
- D2: `dog eats fish` -> [................................................................]
- D3: `cat likes fish` -> [................................................................]

## Exercise 2 - TF

Với D1 có tổng số token là: `........`.

- tf(cat, D1) = `........`
- tf(eats, D1) = `........`
- tf(fish, D1) = `........`
- Tổng TF = `........` (kiểm tra bằng 1)

## Exercise 3 - IDF

Tính theo `idf(t) = log(N / df(t))`, giữ đúng convention của đề.

| term | df | phép tính | IDF |
|---|---:|---|---:|
| cat | 2 | | |
| dog | 1 | | |
| eats | 2 | | |
| fish | 3 | | |
| likes | 1 | | |

Term IDF thấp nhất và lý do: `................................................`

## Exercise 4 - TF-IDF

Tính cho D1 theo `tfidf(t, D1) = tf(t, D1) * idf(t)`.

Giải thích vì sao fish có TF-IDF bằng 0: `................................`

## Exercise 5 - Cosine

Với x = [1,1,1], y = [1,1,0]:

- Tích vô hướng: `........`
- Norm x: `........`
- Norm y: `........`
- Cosine: `........`
- Vì sao không phải 2/3: `................................................`

## Exercise 6 - Dự đoán nhỏ

1. Similarity cao nhất: `........`; vì `...................................`
2. Similarity thấp nhất: `........`; vì `..................................`
3. Term có thể có IDF thấp: `........`; vì `..............................`
4. Nếu bỏ IDF, ranking có đổi không? `....................................`
