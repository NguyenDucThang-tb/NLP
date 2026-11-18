Prediction 1 — Pretrained Transformer và Token Features
Hướng dự đoán: Pretrained Transformer có khả năng đạt Entity-level F1 cao hơn mô hình Logistic Regression sử dụng token features.

Cơ sở lý thuyết:

Transformer sử dụng contextual embeddings, biểu diễn token dựa trên ngữ cảnh của cả câu.

Self-attention giúp mô hình học quan hệ giữa các token, hỗ trợ phân biệt thực thể mơ hồ.

Pretraining cung cấp kiến thức ngôn ngữ từ lượng dữ liệu lớn.

Baseline chủ yếu dựa trên đặc trưng từ vựng, hình thái và ngữ cảnh cục bộ.

Bằng chứng cần kiểm tra: So sánh Precision, Recall và Entity-level F1 của hai mô hình trên cùng test set. Kết quả không nhất thiết luôn nghiêng về Transformer.

Prediction 2 — Entity một token và Entity nhiều token
Hướng dự đoán: Entity nhiều token thường khó nhận diện chính xác hơn entity một token.

Cơ sở lý thuyết:

Entity nhiều token yêu cầu xác định chính xác cả điểm bắt đầu và kết thúc.

Mô hình phải dự đoán đúng chuỗi BIO labels, không chỉ loại thực thể.

Một token bị gán sai có thể khiến toàn bộ entity không được tính là true positive.

Entity dài vẫn có thể dễ nhận diện nếu có cấu trúc quen thuộc và ngữ cảnh rõ ràng.

Bằng chứng cần kiểm tra: So sánh Entity-level F1 theo bốn nhóm độ dài: 1, 2, 3 và 4+ tokens.

Prediction 3 — Entity hiếm
Hướng dự đoán: Entity hiếm trong training data có xu hướng dễ bị nhận diện sai hơn entity xuất hiện thường xuyên.

Cơ sở lý thuyết:

Mô hình có ít ví dụ để học các đặc trưng và ngữ cảnh liên quan đến entity hiếm.

Entity hiếm có thể chứa từ ít gặp hoặc cách viết khác biệt.

Pretrained Transformer có thể giảm khó khăn nhờ subword tokenization và contextual representations.

Tần suất thấp không đồng nghĩa chắc chắn dự đoán sai, vì ngữ cảnh vẫn có thể cung cấp thông tin nhận diện mạnh.

Bằng chứng cần kiểm tra: Thống kê tần suất entity trong training data rồi đối chiếu tỷ lệ nhận diện đúng/sai trên test set theo nhóm tần suất.