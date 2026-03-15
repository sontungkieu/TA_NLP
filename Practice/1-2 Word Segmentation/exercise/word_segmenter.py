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


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in normalize_whitespace(text).split():
        leading: list[str] = []
        trailing: list[str] = []
        core = raw_token

        while core and core[0] in EDGE_PUNCTUATION:
            leading.append(core[0])
            core = core[1:]

        while core and core[-1] in EDGE_PUNCTUATION:
            trailing.append(core[-1])
            core = core[:-1]

        tokens.extend(leading)
        if core:
            tokens.append(core)
        tokens.extend(reversed(trailing))

    return tokens


def load_dictionary(path: str | Path) -> tuple[set[str], int]:
    raise NotImplementedError("TODO: implement dictionary loading and Unicode normalization.")


def segment_text(text: str, lexicon: set[str], max_word_len: int) -> list[str]:
    raise NotImplementedError("TODO: implement forward maximum matching.")


def evaluate(pred_sentences: Iterable[str], gold_sentences: Iterable[str]) -> tuple[float, float, float]:
    raise NotImplementedError("TODO: implement precision, recall, and F1 for word boundaries.")


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
    parser = argparse.ArgumentParser(description="Exercise for Vietnamese word segmentation.")
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
    except NotImplementedError as error:
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
