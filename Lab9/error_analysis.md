# Error analysis evidence — baseline

Các ví dụ dưới đây là ứng viên được tính từ test predictions của baseline. Đây chưa phải lỗi của Transformer. Cột giải thích để trống cho sinh viên tự xem câu và tự phân tích.

Số ứng viên tìm được: {'False positive (non-boundary)': 1300, 'False negative (non-boundary)': 1010, 'Boundary error': 518}

| Sentence ID | Original sentence | Gold span/type/word offsets | Predicted span/type/word offsets | Error category | Student explanation |
|---:|---|---|---|---|---|
| 0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | — | CHINA / LOC / [7, 8) | False positive (non-boundary) | TODO |
| 0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | — | JAPAN / PER / [2, 3) | False positive (non-boundary) | TODO |
| 2 | AL-AIN , United Arab Emirates 1996-12-06 | — | United / ORG / [2, 3) | False positive (non-boundary) | TODO |
| 5 | China controlled most of the match and saw several chances missed until the 78th minute when Uzbek striker Igor Shkvyrin took advantage of a misdirected defensive header to lob the ball over the advancing Chinese keeper and into an empty net . | — | Uzbek / LOC / [16, 17) | False positive (non-boundary) | TODO |
| 7 | The former Soviet republic was playing in an Asian Cup finals tie for the first time . | — | Soviet / LOC / [2, 3) | False positive (non-boundary) | TODO |
| 0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN / LOC / [2, 3) | — | False negative (non-boundary) | TODO |
| 0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | CHINA / PER / [7, 8) | — | False negative (non-boundary) | TODO |
| 5 | China controlled most of the match and saw several chances missed until the 78th minute when Uzbek striker Igor Shkvyrin took advantage of a misdirected defensive header to lob the ball over the advancing Chinese keeper and into an empty net . | Uzbek / MISC / [16, 17) | — | False negative (non-boundary) | TODO |
| 7 | The former Soviet republic was playing in an Asian Cup finals tie for the first time . | Soviet / MISC / [2, 3) | — | False negative (non-boundary) | TODO |
| 12 | Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar 's goal . | Bitar / PER / [27, 28) | — | False negative (non-boundary) | TODO |
| 2 | AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates / LOC / [2, 5) | Arab Emirates / LOC / [3, 5) | Boundary error | TODO |
| 28 | Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | 1995 World Cup / MISC / [6, 9) | World Cup / MISC / [7, 9) | Boundary error | TODO |
| 33 | Squad : Javier Pertile , Paolo Vaccari , Marcello Cuttitta , Ivan Francescato , Leandro Manteri , Diego Dominguez , Francesco Mazzariol , Alessandro Troncon , Orazio Arancio , Andrea Sgorlon , Massimo Giovanelli , Carlo Checchinato , Walter Cristofoletto , Franco Properzi Curti , Carlo Orlandi , Massimo Cuttitta , Giambatista Croci , Gianluca Guidi , Nicola Mazzucato , Alessandro Moscardi , Andrea Castellani . | Franco Properzi Curti / PER / [41, 44) | Franco / PER / [41, 42) | Boundary error | TODO |
| 33 | Squad : Javier Pertile , Paolo Vaccari , Marcello Cuttitta , Ivan Francescato , Leandro Manteri , Diego Dominguez , Francesco Mazzariol , Alessandro Troncon , Orazio Arancio , Andrea Sgorlon , Massimo Giovanelli , Carlo Checchinato , Walter Cristofoletto , Franco Properzi Curti , Carlo Orlandi , Massimo Cuttitta , Giambatista Croci , Gianluca Guidi , Nicola Mazzucato , Alessandro Moscardi , Andrea Castellani . | Franco Properzi Curti / PER / [41, 44) | Curti / PER / [43, 44) | Boundary error | TODO |
| 35 | AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates / LOC / [2, 5) | Arab Emirates / LOC / [3, 5) | Boundary error | TODO |

## Việc sinh viên cần làm

Tự xác nhận từng candidate, chọn ít nhất 5 FP, 5 FN và 5 boundary errors nếu có đủ. Sau đó bổ sung giải thích dựa trên câu gốc. Khi Transformer được fine-tune, chạy lại notebook để có bảng lỗi riêng cho Transformer.
