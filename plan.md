# Plan For Practice 2 Branch Strategy And Deliverables

## Mục tiêu

- Giữ cùng mô hình như `Practice 1`: `main` là nguồn chuẩn cho solution/reference, còn `exercise` là bản giao cho sinh viên.
- Tập trung hiện tại vào `Practice 2.1: POS tagging`; `Practice 2.2: Word embedding` do TA khác làm và sẽ nhập sau.
- Thiết kế `Practice 2.1` sao cho có thể gọi `4` người chính và `1` người dự phòng lên bảng, mỗi người hoàn thiện `1` đoạn code thiên về thuật toán hơn là gọi thư viện.

## Branch Strategy

- Không làm `Practice 2.1` trực tiếp trên branch `exercise`.
- Tạo branch làm việc mới từ `main` với tên:
  - `feat/practice-2-1-pos-tagging`
- Hoàn thiện đầy đủ `Practice 2.1` trên branch này, sau đó merge vào `main`.
- Chỉ sau khi `Practice 2.1` đã ổn trên `main` mới cập nhật branch `exercise` để tạo bản giao sinh viên.
- `Practice 2.2` sẽ đi trên branch riêng của TA còn lại:
  - `feat/practice-2-2-word-embedding`
- Khi `Practice 2.2` hoàn tất, merge vào `main` mà không làm thay đổi cấu trúc đã chốt của `Practice 2.1`.

## Kết quả mong muốn trên `main`

Sau khi hoàn tất `Practice 2.1`, branch `main` cần có thư mục mới theo format giống `Practice 1.2`:

- `Practice/2-1 POS Tagging/README.md`
- `Practice/2-1 POS Tagging/exercise/Practice 2.1 - Exercise.ipynb`
- `Practice/2-1 POS Tagging/exercise/pos_tagger.py`
- `Practice/2-1 POS Tagging/solution/Practice 2.1 - Solution.ipynb`
- `Practice/2-1 POS Tagging/solution/pos_tagger.py`
- `Practice/2-1 POS Tagging/tests/test_pos_tagger.py`
- `Practice/2-1 POS Tagging/docs/feature_design.md`

Quy ước cho `Practice 2.1` trên `main`:

- Không commit toàn bộ Brown corpus vào repo; dữ liệu được tải qua `nltk.download(...)`.
- README của thư mục phải mô tả rõ cách tải Brown corpus, cách train, cách evaluate, cách sinh confusion matrix.
- Notebook và file `.py` phải cùng logic, giống cách tổ chức của `Practice 1.2`.
- Scaffold của exercise và solution phải giống hệt nhau:
  - cùng thứ tự cell trong notebook
  - cùng tên file
  - cùng hàm, cùng chữ ký hàm, cùng CLI arguments
  - cùng phần mô tả, comment và flow chạy
- Khác biệt duy nhất giữa exercise và solution là phần thân các TODO, đáp án mẫu và các cell/đoạn code điền sẵn lời giải.
- Trong bản exercise, mỗi phần TODO phải có ghi chú step-by-step ngắn gọn để sinh viên biết chính xác thứ tự cần làm, nhưng không được lộ trực tiếp đáp án.
- Solution dùng một classifier vừa đủ khó nhưng vẫn dạy được trên bảng; chọn mặc định là `DictVectorizer + LinearSVC`.
- Thư viện xuất hiện ở phần lấy dữ liệu, vector hóa, train classifier và vẽ confusion matrix; phần bài sinh viên vẫn phải tập trung vào biểu diễn đặc trưng, biến đổi dữ liệu, suy diễn và tính metric.

Sau khi TA còn lại hoàn tất `Practice 2.2`, `main` sẽ có thêm:

- `Practice/2-2 Word Embedding/README.md`
- `Practice/2-2 Word Embedding/exercise/...`
- `Practice/2-2 Word Embedding/solution/...`
- `Practice/2-2 Word Embedding/tests/...`

`Practice 2.2` được thêm vào như một thư mục song song, không sửa lại layout của `Practice 2.1`.

## Kết quả mong muốn trên `exercise`

Branch `exercise` chỉ giữ phần giao sinh viên, cùng format như `Practice 1.2` hiện tại:

- `Practice/2-1 POS Tagging/README.md`
- `Practice/2-1 POS Tagging/exercise/Practice 2.1 - Exercise.ipynb`
- `Practice/2-1 POS Tagging/exercise/pos_tagger.py`

Không đưa solution, tests hay docs nội bộ lên `exercise`.

README trên `exercise` chỉ mô tả:

- Brown corpus được tải bằng NLTK.
- Cách mở notebook bài tập.
- Các hàm sinh viên phải hoàn thiện.
- Cách chạy train/evaluate ở mức tối thiểu.
- Tên hàm, tên file, giao diện dòng lệnh và cấu trúc notebook phải khớp tuyệt đối với bản trên `main`.
- Mỗi hàm TODO trong notebook/script phải có checklist step-by-step ngắn, bám đúng flow làm bài.

## Thiết kế bài `Practice 2.1` để gọi 4 `+/-` 1 sinh viên lên bảng

Chọn bài toán POS tagging theo hướng token classification với feature thủ công, vì hướng này cho phép kết hợp thư viện ở mức vừa phải nhưng vẫn giữ trọng tâm ở thuật toán.

Mặc định dùng Brown corpus với `tagset='universal'` cho exercise để số nhãn gọn, dễ giảng và dễ đọc confusion matrix. Trên `main`, solution nên hỗ trợ cả `universal` và `brown` để bài có thêm độ khó và đủ linh hoạt khi cần mở rộng.

Chia phần TODO cho sinh viên thành `4` phần chính và `1` phần mở rộng:

1. `prepare_tagged_sentences(...)`
- Nhận dữ liệu Brown từ NLTK.
- Tách train/test theo danh sách sentence đã cho.
- Chuẩn hóa dữ liệu về danh sách `[(word, tag), ...]`.
- Giữ phần này ở mức biến đổi dữ liệu, không gọi model.
- Ghi chú exercise nên dẫn theo các bước:
  - lấy tagged sentences từ corpus
  - chuẩn hóa từng câu về list cặp `(word, tag)`
  - chia train/test theo tham số
  - trả về đúng cấu trúc dữ liệu mong muốn

2. `extract_token_features(sentence, index, prev_tag=None)`
- Trích đặc trưng cho một token:
  - lowercase word
  - prefix/suffix ngắn
  - có viết hoa hay không
  - có phải chữ số hay không
  - word shape đơn giản như `Xxxx`, `xx`, `dd`
  - token trước/sau
  - tùy chọn `prev_tag`
- Đây là phần chính để sinh viên suy nghĩ về biểu diễn đầu vào.
- Ghi chú exercise nên dẫn theo các bước:
  - xác định token hiện tại
  - tạo feature lexical cơ bản
  - thêm feature ngữ cảnh trái/phải
  - thêm feature phụ thuộc `prev_tag` nếu có
  - trả về một `dict[str, object]`

3. `build_examples(tagged_sentences)` và `predict_sentence(tokens, model, vectorizer)`
- Biến dữ liệu câu có nhãn thành danh sách feature dict và label.
- Khi dự đoán theo câu, dùng greedy decoding nếu có `prev_tag`.
- Lời gọi `vectorizer.fit_transform(...)`, `classifier.fit(...)` chỉ để 1-2 dòng, không biến thành trọng tâm bài.
- Phần khó vừa phải nằm ở chỗ feature của token hiện tại phụ thuộc ngữ cảnh và có thể phụ thuộc `prev_tag`.
- Ghi chú exercise nên dẫn theo các bước:
  - duyệt từng câu và từng token
  - gọi feature extractor để tạo mẫu
  - gom `X` và `y` cho train
  - khi predict thì duyệt trái sang phải
  - cập nhật `prev_tag` bằng nhãn vừa dự đoán

4. `compute_metrics(gold_tags, pred_tags)`
- Tự tính per-tag precision, recall, F1 và macro-F1 từ confusion counts.
- Đây là đoạn thuật toán phù hợp để sinh viên lên bảng hơn là gọi sẵn `classification_report`.
- Ghi chú exercise nên dẫn theo các bước:
  - đếm TP/FP/FN cho từng tag
  - tính precision/recall/F1 cho từng tag
  - lấy trung bình macro
  - trả kết quả theo format đã định

5. Phần mở rộng tùy chọn: `build_baseline_tagger(...)`, `build_confusion_matrix(...)` hoặc `report_top_confusions(...)`
- Tạo baseline đơn giản như `most-frequent-tag` theo từ hoặc liệt kê các cặp tag nhầm nhiều nhất.
- Dùng cho người thứ `5` nếu cần gọi thêm.
- Ghi chú exercise vẫn nên có hướng dẫn step-by-step, nhưng ngắn hơn phần chính để giữ độ mở cho sinh viên khá hơn.

## Public interface cần chốt ngay

`Practice 2.1` nên có một CLI tối giản trong `solution/pos_tagger.py` và bản song song ở `exercise/pos_tagger.py`:

- `--mode train-eval`
- `--tagset {universal,brown}`
- `--test-size 0.2`
- `--seed 42`
- `--confusion-matrix-output <path>`
- `--compare-baseline`

Đầu ra tối thiểu của chế độ `train-eval`:

- per-tag precision/recall/F1
- macro-precision, macro-recall, macro-F1
- confusion matrix lưu thành file ảnh hoặc in bảng rút gọn

Notebook exercise và solution phải dùng cùng pipeline với file `.py`, không tách thành hai hướng cài đặt khác nhau.
Notebook exercise và solution phải có cùng scaffold với file `.py`, không tách thành hai hướng cài đặt khác nhau và không đổi tên hàm giữa hai bản.

## Trình tự triển khai

1. Tạo `feat/practice-2-1-pos-tagging` từ `main`.
2. Dựng đầy đủ `Practice/2-1 POS Tagging/` trên branch này.
3. Hoàn thiện solution trước, sau đó lược bớt thành exercise với các TODO đúng `4 + 1` đoạn.
3. Dùng solution làm scaffold gốc, sau đó sinh exercise bằng cách giữ nguyên scaffold và chỉ thay phần thân lời giải thành TODO đúng `4 + 1` đoạn.
4. Với mỗi TODO trong exercise, thêm ghi chú step-by-step ngay cạnh hàm/cell tương ứng để sinh viên biết thứ tự triển khai.
5. Viết test cho:
- shape dữ liệu đầu ra của feature extractor
- build examples không lệch số lượng token/label
- metric tự tính đúng trên một ví dụ nhỏ
- pipeline train/eval chạy hết không lỗi
- nếu bật baseline thì pipeline trả được cả kết quả baseline và model chính
6. Merge `Practice 2.1` vào `main`.
7. Từ kết quả trên `main`, cập nhật branch `exercise` để chỉ giữ bản giao sinh viên của `Practice 2.1`.
8. Khi `Practice 2.2` hoàn tất từ TA còn lại, merge vào `main` như thư mục độc lập.

## Acceptance criteria

- `main` có `Practice/2-1 POS Tagging/` theo đúng layout giống `Practice 1.2`.
- `exercise` có bản rút gọn chỉ chứa tài liệu và code sinh viên cần.
- Exercise và solution khớp hoàn toàn về scaffold; có thể diff trực tiếp từng cell hoặc từng hàm mà không bị lệch cấu trúc.
- Exercise có hướng dẫn step-by-step đủ rõ để sinh viên biết phải làm gì ở từng TODO, nhưng vẫn phải tự hoàn thiện phần cốt lõi.
- Bài `Practice 2.1` đủ để gọi `4` người chính lên bảng, và có `1` phần dự phòng.
- Nội dung bài thiên về feature engineering, chuyển đổi dữ liệu, suy diễn và tính metric; thư viện chỉ hỗ trợ các phần hợp lý như corpus, vectorizer, classifier và plotting.
- Độ khó đủ nhỉnh hơn `Practice 1.2`: có rich features, có so sánh baseline, và solution trên `main` hỗ trợ cả `universal` lẫn `brown`.
- `Practice 2.2` có thể được thêm sau vào `main` mà không phải đổi branch strategy hay layout của `Practice 2.1`.

## Giả định đã chốt

- Bạn chỉ phụ trách `Practice 2.1`; `Practice 2.2` chưa làm ngay trong branch hiện tại.
- Mô hình branch của repo tiếp tục giống `Practice 1`: `main` cho solution/reference, `exercise` cho bản giao sinh viên.
- Exercise ưu tiên `universal`; solution trên `main` hỗ trợ thêm `brown` để tăng độ khó mà không làm bài trên bảng quá nặng.
- Bản exercise sẽ được tạo từ cùng scaffold của solution, không viết lại một scaffold riêng.
- Root environment sẽ được quản lý bằng `uv`, đồng thời export ra `requirements.txt` để vẫn hỗ trợ `pip install -r requirements.txt`.
- Phần hướng dẫn trong exercise sẽ ở mức procedural step-by-step, không đưa trực tiếp code hoàn chỉnh hay đáp án.
