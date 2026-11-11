# Kiểm tra hoàn thành LAB 08

Mode thực tế: `quick`. Các experiment chưa có EXECUTED không được xem là đã chạy.

|   Bài | File                         | Implemented   | Executed          | Evidence                                                    | Remaining work                                          |
|------:|:-----------------------------|:--------------|:------------------|:------------------------------------------------------------|:--------------------------------------------------------|
|     9 | calculations.md              | template      | STUDENT TODO      | calculations.md                                             | STUDENT TODO: tự tính MLM loss                          |
|    10 | tokenization.ipynb           | IMPLEMENTED   | EXECUTED          | saved notebook output; outputs/verification.json            | STUDENT TODO: ghi và giải thích tokens                  |
|    11 | prediction.md                | template      | STUDENT TODO      | prediction.md                                               | STUDENT TODO: prediction trước experiment               |
|    12 | tokenization.ipynb           | IMPLEMENTED   | EXECUTED          | saved notebook output; outputs/verification.json            | STUDENT TODO: giải thích mask                           |
|    13 | pretrained_classification.py | IMPLEMENTED   | EXECUTED          | saved notebook output; outputs/verification.json            | GPU/T4 và BERT-base NOT_RUN                             |
|    14 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    15 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    16 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    17 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    18 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    19 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED (quick)  | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    20 | error_analysis.md            | IMPLEMENTED   | GENERATED (quick) | outputs/quick/error_candidates.csv                          | STUDENT TODO: error type và hypothesis                  |
|    21 | fine_tuning.ipynb            | IMPLEMENTED   | GENERATED (quick) | outputs/quick/confidence_groups.csv                         | STUDENT TODO: interpretation confidence                 |
|    22 | fine_tuning.ipynb            | IMPLEMENTED   | EXECUTED          | outputs/quick/status.csv; results.csv                       | Full/GPU NOT_RUN; STUDENT TODO: interpretation          |
|    23 | README.md                    | IMPLEMENTED   | GENERATED         | README.md; config.json; environment.json; verification.json | GPU reproducibility NOT_RUN                             |
|    24 | README.md                    | IMPLEMENTED   | GENERATED         | README.md; config.json; environment.json; verification.json | Sinh viên tự rà soát khai báo AI                        |
|    27 | calculations.md              | template      | STUDENT TODO      | calculations.md                                             | STUDENT TODO: 12 câu learning check                     |
|    28 | error_analysis.md            | template      | STUDENT TODO      | error_analysis.md                                           | STUDENT TODO: 6 câu observation/evidence/interpretation |
|    29 | reflection.md                | template      | STUDENT TODO      | reflection.md                                               | STUDENT TODO: reflection 300–500 từ                     |

## Kết quả đã đo

|   experiment_id | initialization   | train_mode   |   train_fraction |   train_samples |   max_length | split      | domain        |   accuracy |   f1_macro |   training_time_seconds |
|----------------:|:-----------------|:-------------|-----------------:|----------------:|-------------:|:-----------|:--------------|-----------:|-----------:|------------------------:|
|              14 | scratch          | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.5   |   0.333333 |                158.362  |
|              14 | pretrained       | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.795 |   0.794749 |                165.12   |
|              15 | pretrained       | frozen       |             1    |             400 |          128 | validation | IMDb          |      0.64  |   0.592944 |                 40.5257 |
|              15 | pretrained       | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.795 |   0.794749 |                165.12   |
|              16 | pretrained       | fine_tuning  |             0.1  |              40 |          128 | validation | IMDb          |      0.55  |   0.539642 |                 17.7404 |
|              16 | pretrained       | fine_tuning  |             0.25 |             100 |          128 | validation | IMDb          |      0.64  |   0.636364 |                 49.8558 |
|              16 | pretrained       | fine_tuning  |             0.5  |             200 |          128 | validation | IMDb          |      0.68  |   0.671593 |                 91.9564 |
|              16 | pretrained       | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.795 |   0.794749 |                165.12   |
|              16 | scratch          | fine_tuning  |             0.1  |              40 |          128 | validation | IMDb          |      0.5   |   0.333333 |                 15.7878 |
|              16 | scratch          | fine_tuning  |             0.25 |             100 |          128 | validation | IMDb          |      0.5   |   0.333333 |                 35.5546 |
|              16 | scratch          | fine_tuning  |             0.5  |             200 |          128 | validation | IMDb          |      0.5   |   0.333333 |                 74.0632 |
|              16 | scratch          | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.5   |   0.333333 |                158.362  |
|              17 | pretrained       | fine_tuning  |             1    |             400 |           64 | validation | IMDb          |      0.725 |   0.722999 |                 82.652  |
|              17 | pretrained       | fine_tuning  |             1    |             400 |          128 | validation | IMDb          |      0.795 |   0.794749 |                165.12   |
|              17 | pretrained       | fine_tuning  |             1    |             400 |          256 | validation | IMDb          |      0.815 |   0.813315 |                243.313  |
|              18 | pretrained       | fine_tuning  |             1    |             400 |          128 | test       | IMDb          |      0.8   |   0.799499 |                165.12   |
|              18 | pretrained       | fine_tuning  |             1    |             400 |          128 | test       | Yelp Polarity |      0.85  |   0.849261 |                165.12   |
|              19 | pretrained       | fine_tuning  |             1    |             400 |          128 | test       | IMDb          |      0.8   |   0.799499 |                165.12   |

CPU peak memory: not measured nếu cột trống. Quick/smoke không thay cho full. Full experiments và GPU/T4 chưa chạy thì vẫn NOT_RUN, dù quick có EXECUTED. Interpretation, calculations, prediction, reflection và giả thuyết lỗi do sinh viên tự hoàn thiện.
