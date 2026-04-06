# TA_NLP-1

Repo này chứa tài liệu và mã nguồn hỗ trợ các buổi practice NLP cho TA.

## Môi trường

Ưu tiên dùng `uv`:

```bash
uv sync
```

Nếu muốn dùng notebook với kernel của môi trường repo:

```bash
uv run python -m ipykernel install --user \
  --name ta-nlp-1 \
  --display-name "Python (ta-nlp-1)"
```

Nếu chỉ dùng `pip`, có thể cài từ file export:

```bash
python3 -m pip install -r requirements.txt
```

Brown corpus sẽ được tải tự động ở lần chạy đầu tiên của `Practice 2.1`, nhưng cũng có thể tải trước:

```bash
uv run python -m nltk.downloader brown universal_tagset
```

## Nội dung hiện có

- `Practice/1-1 Language Model`: notebook cho bài language model.
- `Practice/1-2 Word Segmentation`: exercise, solution, README, test và dữ liệu đánh giá cho word segmentation.
- `Practice/2-1 POS Tagging`: exercise, solution, tài liệu thiết kế đặc trưng và test cho POS tagging với Brown corpus.

## Chạy nhanh Practice 2.1

Solution:

```bash
uv run python "Practice/2-1 POS Tagging/solution/pos_tagger.py" \
  --mode train-eval \
  --tagset universal \
  --compare-baseline \
  --verbose 1 \
  --confusion-matrix-output "Practice/2-1 POS Tagging/solution/universal_confusion_matrix.png"
```

Exercise dùng cùng scaffold và cùng CLI với solution, chỉ khác phần TODO.

## Chạy test

```bash
uv run python -m unittest discover -s "Practice/2-1 POS Tagging/tests"
```
