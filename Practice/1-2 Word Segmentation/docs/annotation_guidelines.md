# Annotation Guidelines

## Nguồn dữ liệu

- Bộ `100` câu được lấy từ `content` của `Practice/1-1 Language Model/train-Subset.txt`.
- Script `scripts/extract_clean_sentences.py` áp dụng bộ lọc để giữ câu trần thuật sạch:
  - kết thúc bằng dấu chấm
  - không có số
  - không có ký tự đặc biệt
  - độ dài vừa phải và không trùng lặp
  - tối đa một câu được lấy từ mỗi bài báo
- File `data/eval_sources.tsv` lưu lại `line_number`, `source_file`, `article_id`, `sentence`.

## Quy ước gold segmentation

- Mỗi dòng tương ứng đúng `1` câu trong `data/eval_input.txt`.
- Token cách nhau bằng đúng `1` khoảng trắng.
- Từ nhiều âm tiết dùng dấu gạch dưới `_`.
- Dấu câu tách riêng nếu nằm ở rìa token, ví dụ `thế_nào ?`.

## Metric

- Precision, recall và F1 được tính theo word boundary.
- Trước khi so sánh, hệ thống bỏ dấu `_` để kiểm tra chuỗi âm tiết gốc giữa prediction và gold có khớp nhau hay không.
