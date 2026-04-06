# Feature Design For Practice 2.1

## Mục tiêu dạy học

Practice này được thiết kế để nghiêng về thuật toán và cách biểu diễn dữ liệu, nhưng vẫn có một phần thư viện vừa đủ để pipeline hoàn chỉnh:

- `nltk`: tải Brown corpus.
- `DictVectorizer`: biến feature dict thành vector.
- `LinearSVC`: train bộ phân loại cho POS tagging.
- `matplotlib`: vẽ confusion matrix.

Trọng tâm không nằm ở việc gọi API thư viện, mà nằm ở:

- chuẩn hóa dữ liệu câu gán nhãn;
- thiết kế feature thủ công cho từng token;
- greedy decoding với `prev_tag`;
- tự tính per-tag precision / recall / F1 và macro-average;
- đọc confusion matrix và phân tích lỗi.

## Feature set mặc định

Feature extractor mặc định của solution dùng:

- `bias`
- `word.lower`
- `word.isupper`
- `word.istitle`
- `word.isdigit`
- `word.has_hyphen`
- `word.length`
- `word.shape`
- `word.prefix1`, `word.prefix2`, `word.prefix3`
- `word.suffix1`, `word.suffix2`, `word.suffix3`
- `BOS`, `EOS`
- `prev.word.lower`, `prev.word.shape`
- `next.word.lower`, `next.word.shape`
- `prev_tag` khi dùng greedy decoding

## Chia phần lên bảng

Phù hợp để gọi `4 +/- 1` sinh viên:

1. `prepare_tagged_sentences(...)`
- tải Brown corpus;
- chuyển về list câu dạng `[(word, tag), ...]`;
- chia train/test tái lập được.

2. `extract_token_features(...)`
- tạo feature cho token hiện tại;
- thêm ngữ cảnh trái/phải;
- thêm `prev_tag` khi có.

3. `build_examples(...)` và `predict_sentence(...)`
- tạo tập mẫu huấn luyện từ câu gán nhãn;
- predict trái sang phải với greedy decoding.

4. `compute_metrics(...)`
- đếm `TP / FP / FN`;
- tính per-tag và macro metric.

5. Mở rộng
- `build_baseline_tagger(...)` hoặc `report_top_confusions(...)`.

## Lý do chọn `universal` cho exercise

Tagset `universal` giúp:

- giảm số nhãn;
- confusion matrix dễ đọc;
- phù hợp với buổi chữa bài trên bảng;
- sinh viên tập trung vào feature engineering và metric thay vì bị quá tải bởi tagset chi tiết.

Solution trên `main` vẫn hỗ trợ thêm `brown` để tăng độ khó và giúp so sánh khi cần.
