from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

DEFAULT_DICT_PATH = Path(__file__).resolve().parents[1] / "vn-dict.txt"
EDGE_PUNCTUATION = set("\"'“”‘’.,!?;:()[]{}|-")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    return " ".join(normalize_unicode(text).split())


def normalize_lookup(text: str) -> str:
    return normalize_whitespace(text).lower()


def split_edge_punctuation(token: str) -> list[str]:
    if not token:
        return []

    leading: list[str] = []
    trailing: list[str] = []
    core = token

    while core and core[0] in EDGE_PUNCTUATION:
        leading.append(core[0])
        core = core[1:]

    while core and core[-1] in EDGE_PUNCTUATION:
        trailing.append(core[-1])
        core = core[:-1]

    parts = leading
    if core:
        parts.append(core)
    parts.extend(reversed(trailing))
    return parts


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in normalize_whitespace(text).split():
        parts = split_edge_punctuation(raw_token)
        tokens.extend(parts if parts else [raw_token])
    return tokens


def load_dictionary(path: str | Path) -> tuple[set[str], int]:
    lexicon: set[str] = set()
    max_word_len = 1

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = normalize_whitespace(line)
            if not entry:
                continue

            lookup_key = entry.lower()
            if lookup_key in lexicon:
                continue

            lexicon.add(lookup_key)
            max_word_len = max(max_word_len, len(lookup_key.split()))

    return lexicon, max_word_len


def segment_tokens(tokens: list[str], lexicon: set[str], max_word_len: int) -> list[str]:
    segmented: list[str] = []
    index = 0

    while index < len(tokens):
        best_match_len = 0
        window = min(max_word_len, len(tokens) - index)

        for size in range(window, 0, -1):
            candidate = " ".join(tokens[index : index + size])
            if normalize_lookup(candidate) in lexicon:
                best_match_len = size
                break

        if best_match_len == 0:
            segmented.append(tokens[index])
            index += 1
            continue

        segmented.append("_".join(tokens[index : index + best_match_len]))
        index += best_match_len

    return segmented


def segment_text(text: str, lexicon: set[str], max_word_len: int) -> list[str]:
    tokens = tokenize_text(text)
    return segment_tokens(tokens, lexicon, max_word_len)


def sentence_to_boundaries(sentence: str) -> tuple[list[str], set[tuple[int, int]]]:
    syllables: list[str] = []
    boundaries: set[tuple[int, int]] = set()
    cursor = 0

    for token in normalize_whitespace(sentence).split():
        parts = token.split("_")
        if any(part == "" for part in parts):
            raise ValueError(f"Invalid token in segmented sentence: {token!r}")

        normalized_parts = [normalize_unicode(part) for part in parts]
        syllables.extend(normalized_parts)
        boundaries.add((cursor, cursor + len(normalized_parts)))
        cursor += len(normalized_parts)

    comparable_syllables = [normalize_lookup(part) for part in syllables]
    return comparable_syllables, boundaries


def evaluate(pred_sentences: Iterable[str], gold_sentences: Iterable[str]) -> tuple[float, float, float]:
    pred_list = list(pred_sentences)
    gold_list = list(gold_sentences)

    if len(pred_list) != len(gold_list):
        raise ValueError("Prediction and gold files must have the same number of lines.")

    matched = 0
    predicted_total = 0
    gold_total = 0

    for line_number, (pred_sentence, gold_sentence) in enumerate(zip(pred_list, gold_list), start=1):
        pred_syllables, pred_boundaries = sentence_to_boundaries(pred_sentence)
        gold_syllables, gold_boundaries = sentence_to_boundaries(gold_sentence)

        if pred_syllables != gold_syllables:
            raise ValueError(
                "Prediction and gold sentence differ after removing segmentation markers "
                f"at line {line_number}."
            )

        matched += len(pred_boundaries & gold_boundaries)
        predicted_total += len(pred_boundaries)
        gold_total += len(gold_boundaries)

    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / gold_total if gold_total else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def run_segment_mode(args: argparse.Namespace) -> int:
    if not args.text and not args.input:
        raise ValueError("Segment mode requires either --text or --input.")

    lexicon, max_word_len = load_dictionary(args.dict)

    if args.text:
        print(" ".join(segment_text(args.text, lexicon, max_word_len)))
        return 0

    predictions = [" ".join(segment_text(line, lexicon, max_word_len)) for line in read_lines(args.input)]

    if args.output:
        write_lines(args.output, predictions)
    else:
        print("\n".join(predictions))

    return 0


def run_evaluate_mode(args: argparse.Namespace) -> int:
    if not args.input or not args.gold:
        raise ValueError("Evaluate mode requires --input and --gold.")

    lexicon, max_word_len = load_dictionary(args.dict)
    predictions = [" ".join(segment_text(line, lexicon, max_word_len)) for line in read_lines(args.input)]

    if args.output:
        write_lines(args.output, predictions)

    precision, recall, f1 = evaluate(predictions, read_lines(args.gold))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vietnamese word segmentation with forward maximum matching.")
    parser.add_argument("--mode", choices=("segment", "evaluate"), required=True)
    parser.add_argument("--dict", default=str(DEFAULT_DICT_PATH), help="Path to the Vietnamese dictionary.")
    parser.add_argument("--text", help="Segment a single sentence from the command line.")
    parser.add_argument("--input", help="Path to the input file. One sentence per line.")
    parser.add_argument("--output", help="Optional path to write predictions.")
    parser.add_argument("--gold", help="Path to the gold segmentation file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "segment":
            return run_segment_mode(args)
        return run_evaluate_mode(args)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
