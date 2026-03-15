from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VIETNAMESE_UPPER = (
    "A-Z"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "Đ"
    "ÈÉẺẼẸÊẾỀỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰ"
    "ỲÝỶỸỴ"
)
SENTENCE_SPLIT_RE = re.compile(rf"(?<=[.!?])\s+(?=[{VIETNAMESE_UPPER}])")
ALLOWED_SENTENCE_RE = re.compile(r"^[A-Za-zÀ-ỹà-ỹ ]+\.$")
BAD_STARTS = (
    "Tuy nhiên",
    "Do đó",
    "Nhưng",
    "Song",
    "Và",
    "Theo",
    "Trong khi đó",
    "Bên cạnh đó",
    "Đồng thời",
    "Ngoài ra",
    "Mới đây",
    "Cũng",
    "Như vậy",
    "Cuối cùng",
    "Hiện tại",
    "Được biết",
)

# Sau bước lọc tự động vẫn còn một số câu méo cấu trúc, trùng hoặc quá phụ thuộc ngữ cảnh.
MANUAL_EXCLUSIONS = {
    "Còn vedette là hoa hậu Trần Tiểu Vy diện thiết kế váy đuôi dài lộng lẫy.",
    "Chúng mình đã đi gần hết các tỉnh thành tại Việt Nam và nuôi giấc mơ du lịch châu Âu cũng nhau.",
    "Sau lớn lên đi làm cũng không biết bao lần khó khăn tìm người tốt và thuyết phục về làm cùng nhau.",
    "Bột đậu lăng Dùng bột đậu lăng đắp mặt sẽ se khít lỗ chân lông.",
    "Vấn đề là bản mệnh có thể gặp phải chút chuyện phát sinh.",
    "Tác dụng Cải thiện bộ nhớ Nhãn có chứa các hợp chất tăng cường chức năng nhận thức và trí nhớ.",
    "Thảm đỏ sự kiện quy tụ nhiều ngôi sao và TikToker đình đám.",
    "Lãi suất tiền gửi nhích lên cũng giúp NH huy động vốn nhiều hơn.",
    "Outfit tưởng không hở mà hở không tưởng của cô làm bao ánh nhìn phải ái ngại.",
    "USD tiếp tục suy yếu và quay về mức thấp nhất trong ba tuần giao dịch gần đây.",
    "Ý nghĩa của cái tên La Casina Delle Civette có nghĩa là Ngôi nhà của những chú cú.",
    "Dmitro nói khi đứng cạnh một khẩu pháo tự hành hướng về phía nam lãnh thổ mà Nga kiểm soát.",
    "Những người chỉ trích cáo buộc vợ tổng thống đang so sánh tương đương các hành động của Kiev và Moscow.",
    "Nga mời chuyên gia LHQ điều tra vụ pháo kích trại tù binh ở Donetsk.",
    "Lái xe đã bị người dân giữ lại.",
}


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def is_clean_sentence(sentence: str) -> bool:
    if sentence in MANUAL_EXCLUSIONS:
        return False

    if not ALLOWED_SENTENCE_RE.fullmatch(sentence):
        return False

    words = sentence[:-1].split()
    if not (8 <= len(words) <= 22):
        return False

    if len({word.lower() for word in words}) < 6:
        return False

    return not any(sentence.startswith(prefix + " ") or sentence == prefix + "." for prefix in BAD_STARTS)


def extract_sentences(content: str) -> list[str]:
    normalized = normalize_whitespace(content)
    return [candidate.strip() for candidate in SENTENCE_SPLIT_RE.split(normalized)]


def select_eval_sentences(records: list[dict], limit: int) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    seen_sentences: set[str] = set()

    for row in records:
        for sentence in extract_sentences(row["content"]):
            if sentence in seen_sentences or not is_clean_sentence(sentence):
                continue

            seen_sentences.add(sentence)
            selected.append((row["id"], sentence))
            break

        if len(selected) >= limit:
            break

    return selected


def write_outputs(selected: list[tuple[int, str]], source_name: str, sentences_out: Path, sources_out: Path) -> None:
    sentences_out.parent.mkdir(parents=True, exist_ok=True)
    sources_out.parent.mkdir(parents=True, exist_ok=True)

    sentences_out.write_text("\n".join(sentence for _, sentence in selected) + "\n", encoding="utf-8")

    with sources_out.open("w", encoding="utf-8") as handle:
        handle.write("line_number\tsource_file\tarticle_id\tsentence\n")
        for line_number, (article_id, sentence) in enumerate(selected, start=1):
            handle.write(f"{line_number}\t{source_name}\t{article_id}\t{sentence}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract clean declarative sentences from train-Subset.txt.")
    parser.add_argument(
        "--input",
        default="Practice/1-1 Language Model/train-Subset.txt",
        help="Path to the JSON corpus.",
    )
    parser.add_argument(
        "--sentences-out",
        default="Practice/1-2 Word Segmentation/data/eval_input.txt",
        help="Output path for selected sentences.",
    )
    parser.add_argument(
        "--sources-out",
        default="Practice/1-2 Word Segmentation/data/eval_sources.tsv",
        help="Output path for provenance metadata.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of sentences to keep.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    selected = select_eval_sentences(records, args.limit)

    if len(selected) < args.limit:
        raise SystemExit(f"Only extracted {len(selected)} clean sentences, fewer than requested {args.limit}.")

    write_outputs(selected, input_path.name, Path(args.sentences_out), Path(args.sources_out))
    print(f"Extracted {len(selected)} sentences from {input_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
