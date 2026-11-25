# Reflection — LAB 10

Qua các lab, em thấy cùng dùng Transformer nhưng đầu ra thay đổi theo nhiệm vụ. Ở LAB 08, mô hình phân loại cảm xúc cho cả review. DistilBERT lấy biểu diễn token đầu, đưa qua classification head để tạo logits cho hai nhãn Negative và Positive. Target của mỗi review là một nhãn duy nhất.

Ở LAB 09, mục tiêu là nhận diện thực thể. Token classification head dự đoán nhãn BIO cho từng token, như `B-PER`, `I-PER` hoặc `O`. Target là chuỗi nhãn căn chỉnh với các từ trong câu. Khi một từ tách thành nhiều subword, phải gán nhãn đúng vị trí và bỏ qua các vị trí không được chấm điểm. Sai căn chỉnh có thể làm hỏng ranh giới thực thể.

LAB 10 làm hỏi đáp trích xuất. Mô hình nhận question và context, rồi dự đoán start logits và end logits cho vị trí trong context. Target là hai biên của câu trả lời, không phải một nhãn cho cả review hay nhãn BIO cho từng token. Vì vậy, ánh xạ character offsets sang token positions và xử lý sliding windows rất quan trọng. Mô hình chọn span trong văn bản, không sinh câu trả lời mới hay truy xuất tài liệu.

Khác biệt chính giữa ba lab là đơn vị dự đoán: toàn review ở LAB 08, từng token ở LAB 09, và hai biên answer span ở LAB 10. Encoder có thể cùng là Transformer, nhưng prediction head, target và cách đánh giá phải phù hợp với bài toán. Với QA, chỉ số token đúng chưa đủ; cần khôi phục đúng đoạn ký tự trong context.

Trong lần kiểm tra hiện tại, notebook đã nạp dữ liệu và tokenizer, đồng thời tạo được nhiều context windows. Tuy nhiên, alignment, span decoding và EM/F1 chưa hoàn thiện nên training chưa chạy. Em chưa có kết quả thực nghiệm để nhận xét model. Sau khi hoàn thiện các hàm và chạy QUICK/FULL, em sẽ bổ sung phân tích dựa trên metrics thật.

**Khai báo sử dụng AI:** Em dùng AI hỗ trợ tạo và chỉnh sửa khung notebook, cấu hình, đường dẫn, thư mục kết quả và kiểm tra trạng thái test. AI cũng hỗ trợ khôi phục `prediction.md` từ nội dung em đã cung cấp và soạn nháp reflection này. AI đã hỗ trợ triển khai alignment, feature preparation, decoding, EM/F1 và phân loại lỗi; em cần đọc, kiểm tra test và hiểu các hàm này trước khi bật training. Chưa có metrics huấn luyện nào được tạo hoặc dùng làm kết luận trong reflection này.
