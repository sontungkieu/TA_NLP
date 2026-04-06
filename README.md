# TA_NLP-1 Exercise Branch

Branch này chỉ giữ các tài liệu và mã nguồn cần giao cho sinh viên.

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

## Nội dung trên branch này

- `Practice/1-1 Language Model`: notebook bài tập cho practice language model.
- `Practice/1-2 Word Segmentation`: bản exercise cho word segmentation.
- `Practice/2-1 POS Tagging`: bản exercise cho POS tagging trên Brown corpus.

## Chạy Practice 2.1

```bash
uv run python "Practice/2-1 POS Tagging/exercise/pos_tagger.py" \
  --mode train-eval \
  --tagset universal \
  --compare-baseline \
  --verbose 1
```

Brown corpus có thể được tải trước bằng:

```bash
uv run python -m nltk.downloader brown universal_tagset
```
