# Practice 2.1: POS Tagging Exercise

Branch này chỉ giữ phần cần giao cho sinh viên:

- `exercise/Practice 2.1 - Exercise.ipynb`: notebook bài tập cho sinh viên.
- `exercise/pos_tagger.py`: skeleton để sinh viên hoàn thiện.

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

## Cách chạy exercise

Notebook:

- Mở `exercise/Practice 2.1 - Exercise.ipynb` nếu muốn làm bài trực tiếp trên Jupyter.
- Notebook trình bày code theo từng cell, cùng scaffold với bản solution ở nhánh `main`.
- Mỗi TODO đều có ghi chú step-by-step để sinh viên biết thứ tự cần làm.

CLI:

```bash
uv run python "Practice/2-1 POS Tagging/exercise/pos_tagger.py" \
  --mode train-eval \
  --tagset universal \
  --compare-baseline \
  --verbose 1
```

## Yêu cầu bài tập

- Cài đặt `prepare_tagged_sentences(...)` để chia Brown corpus thành train/test theo câu.
- Cài đặt `extract_token_features(...)` với feature từ token hiện tại, ngữ cảnh trái/phải và `prev_tag`.
- Cài đặt `build_examples(...)` và `predict_sentence(...)` để train/predict theo pipeline `DictVectorizer + LinearSVC`.
- Cài đặt `compute_metrics(...)` để tự tính per-tag precision, recall, F1 và macro-average.
- Có thể làm thêm `build_baseline_tagger(...)` như phần mở rộng.

## Ghi chú triển khai

- Bản exercise ưu tiên tagset `universal` để số nhãn gọn và phù hợp khi chữa trên bảng.
- Pipeline dùng feature thủ công theo token, greedy decoding theo câu và `LinearSVC`.
- Có thể bật log của `LibLinear` bằng `--verbose 1`.
