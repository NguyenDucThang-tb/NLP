# Reflection

Prediction ban đầu của em về sparsity là đúng. Em nghĩ ma trận sẽ rất sparse vì mỗi document chỉ dùng một phần nhỏ vocabulary. Kết quả có 30.000 documents, vocabulary có 193.837 terms và sparsity là 99,9122%. Tuy nhiên em không dự đoán vocabulary lại lớn như vậy.

Kết quả bất ngờ nhất là search đôi khi trả về document chỉ giống query về mặt từ vựng. Ví dụ query “transformer language model” lại trả về một document nói về transformer/circuit board. Query “natural language processing” cũng trả về các trang có từ language hoặc processing nhưng không thật sự nói về NLP.

Experiment D cung cấp evidence rõ nhất vì các số liệu về kích thước và sparsity được tính trực tiếp trên toàn bộ corpus. Experiment G cũng cho thấy khá rõ giới hạn của TF-IDF khi dùng để search.

Failure case quan trọng nhất là query “transformer language model”. Từ transformer có nhiều nghĩa khác nhau. TF-IDF chỉ nhìn vào các term xuất hiện, không hiểu ngữ cảnh transformer trong điện tử khác với transformer trong xử lý ngôn ngữ. Đây là vấn đề lexical matching và thiếu hiểu biết về ngữ nghĩa.

Nếu xây lại search engine, em sẽ thử bỏ stopword tốt hơn, chuẩn hóa từ và dùng word embedding hoặc mô hình ngữ nghĩa. Em cũng sẽ tạo evaluation labels cẩn thận hơn để so sánh các pipeline bằng P@5, Recall@5 và MRR.

AI contribution:

- AI hỗ trợ tạo khung code TF-IDF, cosine similarity và phần đọc dataset.
- AI hỗ trợ sửa một số lỗi trong notebook và giải thích code.
- Em đã chạy code, xem kết quả, kiểm tra các document và chỉnh sửa nội dung bài.
