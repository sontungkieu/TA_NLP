# Practice 1.2: Word Segmentation

Thư mục này chứa bộ bài làm cho `Practice 1.2` theo yêu cầu:

- `exercise/Practice 1.2 - Exercise.ipynb`: notebook bài tập cho sinh viên.
- `exercise/word_segmenter.py`: skeleton cho sinh viên tự hoàn thiện.
- `solution/Practice 1.2 - Solution.ipynb`: notebook lời giải tham chiếu.
- `solution/word_segmenter.py`: lời giải dùng forward maximum matching.
- `scripts/extract_clean_sentences.py`: script trích 100 câu trần thuật sạch từ `train-Subset.txt`.
- `data/eval_input.txt`: 100 câu đầu vào để đánh giá, lấy từ `content` của `train-Subset.txt`.
- `data/eval_gold.txt`: 100 câu gold segmentation tương ứng.
- `data/eval_sources.tsv`: provenance của 100 câu lấy từ corpora nội bộ.
- `tests/test_word_segmenter.py`: unit test và integration test cho lời giải.
- `docs/annotation_guidelines.md`: quy ước dữ liệu và cách chấm.

## Cách chạy solution

Notebook:

- Mở `exercise/Practice 1.2 - Exercise.ipynb` nếu muốn làm bài trực tiếp trên Jupyter.
- Mở `solution/Practice 1.2 - Solution.ipynb` nếu muốn xem bản hoàn chỉnh cùng logic với file `.py`.

Segment một câu:

```bash
python3 "Practice/1-2 Word Segmentation/solution/word_segmenter.py" \
  --mode segment \
  --text "Bộ Y tế phân tuyến điều trị bệnh đậu mùa khỉ thế nào?"
```

Segment một file:

```bash
python3 "Practice/1-2 Word Segmentation/solution/word_segmenter.py" \
  --mode segment \
  --input "Practice/1-2 Word Segmentation/data/eval_input.txt" \
  --output "Practice/1-2 Word Segmentation/data/eval_pred.txt"
```

Trích lại bộ 100 câu sạch từ `train-Subset.txt`:

```bash
python3 "Practice/1-2 Word Segmentation/scripts/extract_clean_sentences.py"
```

Đánh giá trên bộ 100 câu:

```bash
python3 "Practice/1-2 Word Segmentation/solution/word_segmenter.py" \
  --mode evaluate \
  --input "Practice/1-2 Word Segmentation/data/eval_input.txt" \
  --gold "Practice/1-2 Word Segmentation/data/eval_gold.txt" \
  --output "Practice/1-2 Word Segmentation/data/eval_pred.txt"
```

Chạy test:

```bash
python3 -m unittest discover -s "Practice/1-2 Word Segmentation/tests"
```

## Ghi chú triển khai

- Từ điển được chuẩn hóa Unicode về `NFC`, bỏ dòng trùng và tính lại `max_word_len`.
- Matching dùng lowercase cho lookup nhưng giữ nguyên bề mặt token khi xuất kết quả.
- Khi không tìm được cụm trong từ điển, hệ thống fallback về `1` token đầu vào.
- Gold file dùng `_` để nối các từ nhiều âm tiết; token cách nhau bằng `1` khoảng trắng.
- Bộ `eval_input.txt` được trích từ `content` của `train-Subset.txt`, không dùng `title`.
- Script lọc chỉ giữ câu kết thúc bằng dấu chấm, không có số hay ký tự đặc biệt, và lấy tối đa một câu sạch cho mỗi bài báo.
