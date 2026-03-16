# Practice 1.2: Word Segmentation Exercise

Branch này chỉ giữ phần cần giao cho sinh viên:

- `exercise/Practice 1.2 - Exercise.ipynb`: notebook bài tập cho sinh viên.
- `exercise/word_segmenter.py`: skeleton cho sinh viên tự hoàn thiện.
- `vn-dict.txt`: từ điển tiếng Việt dùng cho maximum matching.
- `data/eval_input.txt`: 100 câu đầu vào lấy trực tiếp từ corpus để thử nghiệm.
- `data/eval_gold.txt`: 100 câu gold segmentation tương ứng để sinh viên tự đánh giá.

## Cách chạy exercise

Notebook:

- Mở `exercise/Practice 1.2 - Exercise.ipynb` nếu muốn làm bài trực tiếp trên Jupyter.
- Cell khởi tạo trong notebook tự dò `base_dir` từ repo root hoặc từ chính thư mục notebook, nên chạy được cả khi kernel mở ở `TA_NLP-1/` hoặc `Practice/1-2 Word Segmentation/exercise/`.
- Notebook exercise và file `exercise/word_segmenter.py` có thêm phần định hướng cho `segment_text(...)` và `evaluate(...)`, nhưng vẫn giữ TODO để sinh viên tự cài đặt.

Segment một câu:

```bash
python3 "Practice/1-2 Word Segmentation/exercise/word_segmenter.py" \
  --mode segment \
  --text "Khẩu súng gây án đã bị thu giữ."
```

Segment một file:

```bash
python3 "Practice/1-2 Word Segmentation/exercise/word_segmenter.py" \
  --mode segment \
  --input "Practice/1-2 Word Segmentation/data/eval_input.txt"
```

Tự chấm trên bộ 100 câu:

```bash
python3 "Practice/1-2 Word Segmentation/exercise/word_segmenter.py" \
  --mode evaluate \
  --input "Practice/1-2 Word Segmentation/data/eval_input.txt" \
  --gold "Practice/1-2 Word Segmentation/data/eval_gold.txt"
```

## Yêu cầu bài tập

- Cài đặt `load_dictionary(path) -> lexicon, max_word_len`.
- Cài đặt `segment_text(text, lexicon, max_word_len) -> list[str]` bằng forward maximum matching.
- Cài đặt `evaluate(pred_sentences, gold_sentences) -> precision, recall, f1` nếu muốn tự chấm thêm.
- Chuẩn hóa Unicode nhất quán trước khi tra từ điển.
- Khi không match được, fallback về `1` token đầu vào.
