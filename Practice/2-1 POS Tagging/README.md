# Practice 2.1: POS Tagging

Thư mục này chứa bộ bài làm cho `Practice 2.1` theo đúng format của `Practice 1.2`:

- `exercise/Practice 2.1 - Exercise.ipynb`: notebook bài tập cho sinh viên, có code theo từng cell song song với file `.py`.
- `exercise/pos_tagger.py`: skeleton để sinh viên hoàn thiện.
- `solution/Practice 2.1 - Solution.ipynb`: notebook lời giải tham chiếu, có code theo từng cell song song với file `.py`.
- `solution/pos_tagger.py`: lời giải đầy đủ dùng Brown corpus, `DictVectorizer` và `LinearSVC`.
- `tests/test_pos_tagger.py`: test cho feature extractor, metric và pipeline nhỏ.
- `docs/feature_design.md`: mô tả feature set và cách chia phần lên bảng.

## Cách cài môi trường

Ưu tiên dùng `uv`:

```bash
uv sync
```

Nếu muốn dùng notebook với kernel riêng của repo:

```bash
uv run python -m ipykernel install --user \
  --name ta-nlp-1 \
  --display-name "Python (ta-nlp-1)"
```

Nếu chỉ dùng `pip`:

```bash
python3 -m pip install -r requirements.txt
```

Brown corpus sẽ được tải tự động khi chạy lần đầu. Nếu muốn tải trước:

```bash
uv run python -m nltk.downloader brown universal_tagset
```

## Cách chạy solution

Evaluate với tagset `universal` và so sánh baseline:

```bash
uv run python "Practice/2-1 POS Tagging/solution/pos_tagger.py" \
  --mode train-eval \
  --tagset universal \
  --compare-baseline \
  --verbose 1 \
  --confusion-matrix-output "Practice/2-1 POS Tagging/solution/universal_confusion_matrix.png"
```

Evaluate với tagset Brown gốc:

```bash
uv run python "Practice/2-1 POS Tagging/solution/pos_tagger.py" \
  --mode train-eval \
  --tagset brown \
  --test-size 0.2 \
  --seed 42 \
  --verbose 1 \
  --confusion-matrix-output "Practice/2-1 POS Tagging/solution/brown_confusion_matrix.png"
```

## Cách chạy exercise

Exercise và solution dùng cùng scaffold:

- cùng tên file
- cùng hàm
- cùng chữ ký hàm
- cùng CLI
- cùng thứ tự cell notebook
- cùng các phần code chính được tách trực tiếp thành từng cell trong notebook

Khác biệt duy nhất là bản exercise để trống các đoạn TODO và có ghi chú step-by-step để sinh viên biết trình tự cần làm.
Notebook không chỉ gọi lại module `.py`, mà còn trình bày cùng pipeline dưới dạng từng cell để tiện chữa bài trực tiếp trên lớp.

Chạy exercise:

```bash
uv run python "Practice/2-1 POS Tagging/exercise/pos_tagger.py" \
  --mode train-eval \
  --tagset universal \
  --verbose 1 \
  --compare-baseline
```

## Ghi chú triển khai

- Brown corpus được chia train/test theo câu với `test_size=0.2` và `seed=42` mặc định.
- Bản exercise ưu tiên tagset `universal` để số nhãn gọn và phù hợp khi chữa trên bảng.
- Bản solution hỗ trợ cả `universal` và `brown`.
- Pipeline dùng feature thủ công theo token, greedy decoding theo câu và `LinearSVC` cho classifier.
- Có thể bật log của `LibLinear` bằng `--verbose 1` để xem quá trình tối ưu khi train `LinearSVC`.
- Metric được tính thủ công theo từng tag và macro-average, không phụ thuộc `classification_report`.
- Confusion matrix được tính thủ công rồi vẽ bằng `matplotlib`.
