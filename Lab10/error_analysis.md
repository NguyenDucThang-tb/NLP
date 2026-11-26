# Error analysis — LAB 10

## Nguồn và phạm vi

Phân tích dưới đây dùng `outputs/full_seed42/error_candidates.csv`, gồm 1.000 mẫu held-out của SQuAD được đánh giá với DistilBERT QA, `max_length=384`, `stride=64`, seed 42. Các vị trí là character offsets trong context gốc. Bảng ví dụ được chọn trực tiếp từ CSV; phần giải thích ghi nhận điều thấy được trong câu và nêu nguyên nhân như giả thuyết, không khẳng định mô hình đã dựa vào một dấu hiệu cụ thể.

## 1. Wrong start — 5 ví dụ

| Sample ID | Question | Gold | Prediction | Error type | Quan sát từ context / giả thuyết cần kiểm tra |
|---|---|---|---|---|---|
| `572838323acd2414000df73b` | What award was Michelle Gomez nominated for? | Best Supporting Actress | BAFTA Television Award for Best Actor and in 2016, Michelle Gomez became the first female to receive a BAFTA nomination for the series, getting a Best Supporting Actress | wrong_start_only | End offset trùng gold nhưng prediction lấy thêm phần lớn câu dẫn trước đáp án. Có thể span dài đã được chọn thay vì cụm danh từ ngắn; đây là nhận xét từ span, chưa xác định nguyên nhân nội bộ. |
| `5711658e50c2381900b54ad8` | Other than warships, what ships typically required high speeds? | ocean liners | dreadnought battleships, and ocean liners | wrong_start_only | Prediction thêm “dreadnought battleships, and” ở trước đáp án trong cùng danh sách. Có thể mô hình lấy rộng cả phần liệt kê. |
| `5705fd8475f01819005e7844` | The mountain ranges tail off into what kind of geographical formation? | valleys | large and small interior valleys | wrong_start_only | Gold nằm ở cuối cụm “large and small interior valleys”; prediction lấy cả phần mô tả đứng trước. |
| `572fdb17b2c2fd1400568521` | What act sets the term for judging the boundaries of sanity to which individuals wishing to sit on the SP must adhere? | Mental Health (Care and Treatment) (Scotland) Act 2003 | the Mental Health (Care and Treatment) (Scotland) Act 2003 | wrong_start_only | Prediction chỉ thêm mạo từ “the”; F1 trong CSV là 100 dù boundary start khác. Đây là trường hợp span khác ranh giới nhưng nội dung trả lời tương đương. |
| `56e0ff5dcd28a01900c673fc` | What happened to the dynamos in the power station? | repeatedly burned out | burned out | wrong_start_only | Prediction bỏ từ “repeatedly” ở đầu đáp án, còn end offset đúng. Có thể mô hình chọn cụm ngắn hơn trong câu. |

## 2. Wrong end — 5 ví dụ

| Sample ID | Question | Gold | Prediction | Error type | Quan sát từ context / giả thuyết cần kiểm tra |
|---|---|---|---|---|---|
| `5729f799af94a219006aa70a` | Long-lived memory cells can remember previous encounters with what? | specific pathogen | specific pathogen encountered | wrong_end_only | Start offset đúng nhưng prediction thêm “encountered”. Có thể ranh giới đáp án được đặt sau động từ theo cấu trúc câu thay vì kết thúc ở danh từ được hỏi. |
| `572a03086aef0514001551a3` | Low doses of anti-inflammatories are sometimes used with what classes of drugs? | cytotoxic or immunosuppressive drugs | cytotoxic or immunosuppressive drugs such as methotrexate or azathioprine | wrong_end_only | Prediction giữ nguyên cụm gold rồi thêm ví dụ về hai loại thuốc. Có thể span đã mở rộng sang phần giải thích tiếp theo. |
| `5733f410d058e614000b6665` | How did King Louis XV respond to British plans? | dispatched six regiments to New France under the command of Baron Dieskau in 1755. | dispatched six regiments to New France under the command of Baron Dieskau in 1755. The British, intending to blockade French ports | wrong_end_only | Start offset trùng gold; phần dư bắt đầu ngay ở câu tiếp theo. Có thể mô hình đã không dừng span tại dấu kết thúc câu. |
| `5705e33f52bb89140068964f` | Which border does the megaregion extend over? | Mexican | Mexican border into Tijuana | wrong_end_only | Prediction tiếp tục qua “border into Tijuana” sau từ “Mexican”. Có thể mô hình chọn cả cụm danh từ thay vì tính từ làm đáp án. |
| `5725c743ec44d21400f3d549` | What are the two primary constitutional sources of the European Union? | Treaty on European Union (TEU) and the Treaty on the Functioning of the European Union (TFEU) | Treaty on European Union (TEU) | wrong_end_only | Prediction chỉ lấy nguồn thứ nhất trong khi gold chứa hai nguồn nối bằng “and”. Có thể đáp án dài hoặc cấu trúc liệt kê làm ranh giới bị cắt sớm. |

## 3. Wrong span — 5 ví dụ không overlap

| Sample ID | Question | Gold | Prediction | Error type | Quan sát từ context / giả thuyết cần kiểm tra |
|---|---|---|---|---|---|
| `56f8c43d9b226e1400dd0f64` | Who scolded Luther about his rudeness? | His wife Katharina | Justus Jonas | non_overlapping | Gold xuất hiện trong câu nói về vợ Luther; prediction là tên người trong đoạn sau nói Luther ở lại nhà bạn. Có thể tên riêng khác trong context gây nhầm, nhưng offset chỉ xác nhận mô hình chọn sai người. |
| `5728855d3acd2414000dfa90` | Why were the Uighurs ranked higher by the Mongols? | the Uighurs surrendered peacefully without violently resisting | last | non_overlapping | Gold nêu lý do đầu hàng hòa bình; “last” nằm trong phần nói về thứ tự đầu hàng của người Koreans. Có thể mô hình bị hút bởi thông tin gần chủ đề nhưng trả lời thuộc mệnh đề khác. |
| `570d4030fed7b91900d45da0` | Which rail line operates in Melbourne? | Metro Trains Melbourne | V/Line | non_overlapping | Context liệt kê nhiều nhà khai thác; gold là Metro Trains Melbourne, prediction chọn V/Line ở phần tiếp theo. Đây là lỗi chọn một mục khác trong danh sách. |
| `572a135daf94a219006aa7a2` | What is the ability to recognize and adapt to new specific pathogens called? | Adaptive (or acquired) immunity | over time to recognize specific pathogens more efficiently | non_overlapping | Prediction chép phần mô tả chức năng xuất hiện ngay trước tên “Adaptive (or acquired) immunity”, thay vì thuật ngữ được hỏi. Có thể câu hỏi đòi tên khái niệm nhưng hai đoạn có từ vựng gần nhau. |
| `57283dbeff5b5019007d9fc9` | What Doctor Who actor won a Best Actor award in 2012? | Matt Smith | The Doctor's Wife | non_overlapping | “The Doctor's Wife” là tên tập xuất hiện gần câu có Matt Smith; prediction lấy tên tác phẩm thay cho tên diễn viên. Có thể nhiều thực thể cùng xuất hiện trong đoạn làm tăng nhầm lẫn. |

## 4. Thống kê wrong start và wrong end

Đếm trên toàn bộ 1.000 dòng trong `error_candidates.csv`, so sánh trực tiếp predicted và gold character offsets:

| Kết quả | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Span đúng (`correct`) | 648 | 64.8% |
| Chỉ sai start (`wrong_start_only`) | 103 | 10.3% |
| Chỉ sai end (`wrong_end_only`) | 74 | 7.4% |
| Sai cả hai boundary nhưng còn overlap (`both_boundaries_wrong`) | 12 | 1.2% |
| Không overlap (`non_overlapping`) | 163 | 16.3% |
| Start offset khác gold, gồm các nhóm lỗi có chạm start | 278 | 27.8% |
| End offset khác gold, gồm các nhóm lỗi có chạm end | 249 | 24.9% |
| Span khác gold (`wrong_span`) | 352 | 35.2% |

Các nhóm `wrong_start` và `wrong_end` có thể cùng tính một mẫu nếu cả hai đầu sai; vì vậy không cộng hai tỷ lệ này để suy ra tổng số mẫu lỗi. Phân loại `error_type` chia 1.000 mẫu thành các nhóm không chồng lặp.

## 5. Các dạng lỗi quan sát được

- **Wrong boundary / span quá rộng hoặc quá hẹp:** ví dụ “the” bị thêm vào tên đạo luật, “repeatedly” bị bỏ, hoặc prediction kéo dài qua phần ví dụ/câu kế tiếp.
- **Partial answer:** câu hỏi về hai nguồn hiến pháp chỉ được trả lời bằng nguồn thứ nhất.
- **Chọn distractor trong cùng context:** V/Line thay cho Metro Trains Melbourne; “The Doctor's Wife” thay cho Matt Smith.
- **Cụm mô tả gần với đáp án:** mô hình chọn phần mô tả “over time to recognize…” thay vì tên khái niệm “Adaptive immunity”.
- **Đáp án dài:** lỗi có đáp án dài cần được kiểm tra riêng; trong CSV ví dụ về câu hỏi hai nguồn có prediction bị cắt. Đây chưa đủ để kết luận độ dài là nguyên nhân chung.
- **Truncation/tokenization:** bảng offsets cuối cùng không chứng minh đây là nguyên nhân. Cần kiểm tra feature window, offsets và token positions của từng mẫu trước khi kết luận.

## 6. Giả thuyết và giới hạn

**Quan sát:** trong 1.000 mẫu, 352 mẫu có span khác gold theo phân loại CSV; 163 mẫu không overlap. Sai start xảy ra ở 278 mẫu, sai end ở 249 mẫu. Một số lỗi là boundary nhỏ nhưng đáp án gần như tương đương; một số khác chọn thực thể hoặc mệnh đề khác trong context.

**Giả thuyết cần kiểm chứng:** các danh sách thực thể, nhiều câu trả lời liên quan trong cùng đoạn, và đáp án gồm nhiều thành phần có thể làm việc chọn ranh giới hoặc chọn span khó hơn. Các ví dụ trên chỉ gợi ý giả thuyết; chúng không cho biết attention/score nội bộ hay chứng minh nguyên nhân gây lỗi.

**Giới hạn:** đây là tập 1.000 mẫu held-out trong cấu hình `max_length=384`, `stride=64`, không phải phân tích toàn bộ SQuAD. CSV có character offsets và câu trả lời để phân loại hình dạng lỗi, nhưng muốn kết luận về distractor, truncation, tokenization hoặc insufficient context cần kiểm tra thêm feature-level evidence và đối chiếu thủ công. Kết quả tổng thể cùng cấu hình trong `results.csv` là EM 68.0 và token F1 77.4541; F1 trung bình trong `error_candidates.csv` khớp giá trị này.
