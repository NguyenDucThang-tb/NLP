# Phân tích lỗi — Các ví dụ trên tập kiểm tra của LSTM

Nguồn: tập kiểm tra IMDb; các mẫu được chọn dựa trên dự đoán của LSTM. Nhãn: 0 = Tiêu cực, 1 = Tích cực.
Các dấu hiệu tự động phát hiện được chỉ là đặc điểm quan sát thấy, không phải nguyên nhân đã được xác nhận.

## Mẫu kiểm tra 12500 — Dự đoán đúng

- Nhãn thực tế: 1
- Dự đoán của LSTM: 1
- P(positive): 0.514119
- Độ dài chuỗi: 142; số token được giữ lại: 100
- Bị cắt: 42 token (29.58%)
- Token UNK: 3 (3.00% số token được giữ lại)
- Dấu hiệu phủ định: không phát hiện
- Từ nối tương phản: `but`
- Dự đoán RNN / P(+): 1 / 0.538806
- Dự đoán GRU / P(+): 1 / 0.896044
- Trích đoạn đánh giá:

> Previous reviewer Claudio Carvalho gave a much better recap of the film's plot details than I could. What I recall mostly is that it was just so beautiful, in every sense - emotionally, visually, editorially - just gorgeous.<br /><br />If you like movies that are wonderful to look at, and also have emotional content to which that beauty is relevant, I think you will be glad to have seen this extraordinary and unusual work of art.<br /><br />On a scale of 1 to 10, I'd give it about an 8.75. The only reason I shy away from 9 is that it is a mood piece. If you are in the mood for a really artistic, very romantic film, then it's a 10. I definitely think it's a must-see, but none of us can be in that mood all the time, so, overall, 8.75.

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]

## Mẫu kiểm tra 2 — Dự đoán đúng

- Nhãn thực tế: 0
- Dự đoán của LSTM: 0
- P(positive): 0.383809
- Độ dài chuỗi: 134; số token được giữ lại: 100
- Bị cắt: 34 token (25.37%)
- Token UNK: 2 (2.00% số token được giữ lại)
- Dấu hiệu phủ định: `not`
- Từ nối tương phản: `though`
- Dự đoán RNN / P(+): 1 / 0.555598
- Dự đoán GRU / P(+): 0 / 0.114278
- Trích đoạn đánh giá:

> its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]

## Mẫu kiểm tra 3 — Dự đoán đúng

- Nhãn thực tế: 0
- Dự đoán của LSTM: 0
- P(positive): 0.429366
- Độ dài chuỗi: 355; số token được giữ lại: 100
- Bị cắt: 255 token (71.83%)
- Token UNK: 5 (5.00% số token được giữ lại)
- Dấu hiệu phủ định: `never`, `no`
- Từ nối tương phản: `but`
- Dự đoán RNN / P(+): 0 / 0.463880
- Dự đoán GRU / P(+): 0 / 0.118782
- Trích đoạn đánh giá:

> STAR RATING: ***** Saturday Night **** Friday Night *** Friday Morning ** Sunday Night * Monday Morning <br /><br />Former New Orleans homicide cop Jack Robideaux (Jean Claude Van Damme) is re-assigned to Columbus, a small but violent town in Mexico to help the police there with their efforts to stop a major heroin smuggling operation into their town. The culprits turn out to be ex-military, lead by former commander Benjamin Meyers (Stephen Lord, otherwise known as Jase from East Enders) who is using a special method he learned in Afghanistan to fight off his opponents. But Jack has a more personal reason for taking him down, that draws the two men into an explosive final showdown where only one will walk away alive.<br /><br />After Until Death, Van Damme appeared to be on a high, showing he could make the best straight to video films in the action market. While that was a far more drama oriented film, with The Shepherd he has returned to the high-kicking, no brainer action that first made him famous and has sadly produced his worst film since Derailed. It's nowhere near as bad as that film, but what I said still stands.<br /><br />A dull, predictable film, with very little in the

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]

## Mẫu kiểm tra 0 — Dự đoán sai

- Nhãn thực tế: 0
- Dự đoán của LSTM: 1
- P(positive): 0.521517
- Độ dài chuỗi: 240; số token được giữ lại: 100
- Bị cắt: 140 token (58.33%)
- Token UNK: 2 (2.00% số token được giữ lại)
- Dấu hiệu phủ định: `not`
- Từ nối tương phản: `but`, `yet`
- Dự đoán RNN / P(+): 1 / 0.584997
- Dự đoán GRU / P(+): 0 / 0.435017
- Trích đoạn đánh giá:

> I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert brea

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]

## Mẫu kiểm tra 12505 — Dự đoán sai

- Nhãn thực tế: 1
- Dự đoán của LSTM: 0
- P(positive): 0.461034
- Độ dài chuỗi: 353; số token được giữ lại: 100
- Bị cắt: 253 token (71.67%)
- Token UNK: 2 (2.00% số token được giữ lại)
- Dấu hiệu phủ định: `not`
- Từ nối tương phản: `but`
- Dự đoán RNN / P(+): 0 / 0.436573
- Dự đoán GRU / P(+): 1 / 0.800630
- Trích đoạn đánh giá:

> Overall, a well done movie. There were the parts that made me wince, and there were the parts that I threw my hands up at, but I came away with something more than I gone in with.<br /><br />I think the movie suffers from some serious excess ambition. Without spoiling it, let me say that the obvious references to the trial by fire in Ramayana, is way beyond what this movie stands for. The Ramayana is an epic. Not a 200 page book that puts down women in India. The movie is about two girls married into a very distinctive Indian family. While the basic tenets of the "unwritten laws of the family tradition" seem to be that of conservative India, let me assure my reader that I (having lived in Delhi for 12 years) found entire parts that just did not ring those bells. I mean some things and some actions are very true, but some other stuff is just way off the mark. Especially today.<br /><br />Delhi is complicated. India is complicated. The director tries to simplify both. And fails pretty miserably at that.<br /><br />Why in English? Can you imagine a movie about American Indians in English. Or the French speaking in English. Seriously jarring. Even the servant spoke in fluent accented E

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]

## Mẫu kiểm tra 571 — Dự đoán sai

- Nhãn thực tế: 0
- Dự đoán của LSTM: 1
- P(positive): 0.751837
- Độ dài chuỗi: 156; số token được giữ lại: 100
- Bị cắt: 56 token (35.90%)
- Token UNK: 4 (4.00% số token được giữ lại)
- Dấu hiệu phủ định: không phát hiện
- Từ nối tương phản: `but`, `though`
- Dự đoán RNN / P(+): 1 / 0.594503
- Dự đoán GRU / P(+): 1 / 0.803270
- Trích đoạn đánh giá:

> Dorothy Provine does the opposite here: She keeps growing and growing. I didn't detect any subtext, though. "The Incredible Shrinking Man" and other movies of its ilk during the period were parables about radiation, nuclear war, and other horrors. Provine's growth is the result of an inept computer/robot.<br /><br />And who operates this computer but Lou Costello! I like some of his movies with Bud Abbott. But, though this is a pretty bad movie, he does fine without him. And Gale Green is an excellent foil.<br /><br />Green plays the pompous town big shot. He is Provine's father. He is intent on being elected Mayor. So when his beloved daughter starts having issues, he dumps her. He doesn't exactly dump her but gives up his battle against her longtime admirer Costello.<br /><br />This is pretty implausible: Costello is the local garbage collector.<br /><br />The special effects are minimal. And the subplot involving the military is lame in the extreme.

- Bằng chứng:
  - Dựa vào các số liệu ở trên và xem toàn văn trong tệp CSV.
- Nguyên nhân có thể có: đây chỉ là giả thuyết; cần kiểm tra với nội dung đánh giá và chuỗi đầu vào thực tế.
- Đây có phải lỗi do mô hình hóa chuỗi gây ra không? Cần thêm bằng chứng nào để xác định?
- Phân tích của sinh viên: [tự viết]
