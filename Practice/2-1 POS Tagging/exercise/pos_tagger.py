from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

VALID_TAGSETS = ("universal", "brown")


def ensure_brown_corpus(tagset: str = "universal") -> None:
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)

    if tagset == "universal":
        try:
            nltk.data.find("taggers/universal_tagset")
        except LookupError:
            nltk.download("universal_tagset", quiet=True)


def load_brown_sentences(tagset: str = "universal") -> list[list[tuple[str, str]]]:
    if tagset not in VALID_TAGSETS:
        raise ValueError(f"Unsupported tagset: {tagset!r}. Choose from {VALID_TAGSETS}.")

    ensure_brown_corpus(tagset)
    corpus_tagset = "universal" if tagset == "universal" else None
    raw_sentences = brown.tagged_sents(tagset=corpus_tagset)

    tagged_sentences: list[list[tuple[str, str]]] = []
    for sentence in raw_sentences:
        normalized_sentence = [(word, tag) for word, tag in sentence if word and tag]
        if normalized_sentence:
            tagged_sentences.append(normalized_sentence)

    return tagged_sentences


def prepare_tagged_sentences(
    tagset: str = "universal",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[tuple[str, str]]], list[list[tuple[str, str]]]]:
    # Step-by-step:
    # 1. Validate that test_size is strictly between 0 and 1.
    # 2. Load Brown tagged sentences with load_brown_sentences(tagset).
    # 3. Build a reproducible list of sentence indices using random.Random(seed).
    # 4. Decide how many sentences go to the test split.
    # 5. Return (train_sentences, test_sentences) with the same sentence format.
    raise NotImplementedError("TODO: prepare Brown train/test splits.")


def word_shape(token: str) -> str:
    shape: list[str] = []
    for character in token:
        if character.isupper():
            shape.append("X")
        elif character.islower():
            shape.append("x")
        elif character.isdigit():
            shape.append("d")
        else:
            shape.append(character)
    return "".join(shape)


def extract_token_features(
    sentence: Sequence[str],
    index: int,
    prev_tag: str | None = None,
) -> dict[str, object]:
    # Step-by-step:
    # 1. Read the current token with sentence[index].
    # 2. Build lexical features such as lowercase, prefixes, suffixes, shape and length.
    # 3. Add sentence-position flags like BOS / EOS.
    # 4. Add left and right context features from the neighboring tokens.
    # 5. If prev_tag is available, include it as an extra feature.
    # 6. Return a single dict[str, object] for DictVectorizer.
    raise NotImplementedError("TODO: extract token-level features.")


def build_examples(tagged_sentences: Iterable[Sequence[tuple[str, str]]]) -> tuple[list[dict[str, object]], list[str]]:
    # Step-by-step:
    # 1. Loop over each tagged sentence.
    # 2. Split each sentence into a token list and a gold-tag list.
    # 3. For each token, call extract_token_features(...).
    # 4. Use the previous gold tag during training when available.
    # 5. Collect one feature dict in X and one gold tag in y for every token.
    raise NotImplementedError("TODO: convert tagged sentences into supervised examples.")


def train_classifier(
    train_sentences: Iterable[Sequence[tuple[str, str]]],
    verbose: int = 0,
) -> tuple[DictVectorizer, LinearSVC, str]:
    feature_rows, labels = build_examples(train_sentences)
    if not feature_rows:
        raise ValueError("Training split is empty; cannot train the classifier.")

    vectorizer = DictVectorizer(sparse=True)
    classifier = LinearSVC(dual="auto", random_state=0, max_iter=5000, verbose=verbose)
    design_matrix = vectorizer.fit_transform(feature_rows)
    classifier.fit(design_matrix, labels)
    default_tag = Counter(labels).most_common(1)[0][0]
    return vectorizer, classifier, default_tag


def predict_sentence(
    tokens: Sequence[str],
    classifier: LinearSVC,
    vectorizer: DictVectorizer,
    default_tag: str,
) -> list[str]:
    # Step-by-step:
    # 1. Predict tags from left to right over the token sequence.
    # 2. Start with prev_tag = None for the first token.
    # 3. At each position, extract features for the current token.
    # 4. Vectorize that single feature dict and ask the classifier for one prediction.
    # 5. Append the predicted tag and reuse it as prev_tag for the next token.
    # 6. Return the full predicted tag sequence.
    raise NotImplementedError("TODO: decode one sentence with greedy left-to-right prediction.")


def predict_tagged_sentences(
    tagged_sentences: Iterable[Sequence[tuple[str, str]]],
    classifier: LinearSVC,
    vectorizer: DictVectorizer,
    default_tag: str,
) -> list[list[str]]:
    predicted_sentences: list[list[str]] = []
    for sentence in tagged_sentences:
        tokens = [word for word, _ in sentence]
        predicted_sentences.append(predict_sentence(tokens, classifier, vectorizer, default_tag))
    return predicted_sentences


def flatten_gold_tags(tagged_sentences: Iterable[Sequence[tuple[str, str]]]) -> list[str]:
    return [tag for sentence in tagged_sentences for _, tag in sentence]


def flatten_predicted_tags(predicted_sentences: Iterable[Sequence[str]]) -> list[str]:
    return [tag for sentence in predicted_sentences for tag in sentence]


def build_baseline_tagger(train_sentences: Iterable[Sequence[tuple[str, str]]]) -> tuple[dict[str, str], str]:
    # Optional extension:
    # 1. Count tags for each lowercase word type.
    # 2. Keep the most frequent tag of each word as the baseline prediction.
    # 3. Keep the most frequent overall tag as the fallback for unknown words.
    raise NotImplementedError("TODO: implement the optional most-frequent-tag baseline.")


def predict_with_baseline(
    tagged_sentences: Iterable[Sequence[tuple[str, str]]],
    baseline_map: dict[str, str],
    default_tag: str,
) -> list[list[str]]:
    predictions: list[list[str]] = []
    for sentence in tagged_sentences:
        sentence_predictions = [baseline_map.get(word.lower(), default_tag) for word, _ in sentence]
        predictions.append(sentence_predictions)
    return predictions


def compute_metrics(gold_tags: Sequence[str], pred_tags: Sequence[str]) -> dict[str, object]:
    # Step-by-step:
    # 1. Check that gold_tags and pred_tags have the same length.
    # 2. Build the full label set from gold and prediction.
    # 3. For each tag, compute TP, FP, FN and support.
    # 4. Convert those counts into precision, recall and F1.
    # 5. Average the per-tag scores to get macro precision, recall and macro F1.
    # 6. Return a dict that keeps labels, per_tag scores and macro scores together.
    raise NotImplementedError("TODO: compute per-tag and macro metrics.")


def build_confusion_matrix(
    gold_tags: Sequence[str],
    pred_tags: Sequence[str],
    labels: Sequence[str],
) -> list[list[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for gold, pred in zip(gold_tags, pred_tags):
        matrix[label_to_index[gold]][label_to_index[pred]] += 1

    return matrix


def report_top_confusions(
    confusion_matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    limit: int = 10,
) -> list[tuple[str, str, int]]:
    ranked_confusions: list[tuple[str, str, int]] = []
    for gold_index, gold_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if gold_index == pred_index:
                continue
            count = confusion_matrix[gold_index][pred_index]
            if count:
                ranked_confusions.append((gold_label, pred_label, count))

    ranked_confusions.sort(key=lambda item: item[2], reverse=True)
    return ranked_confusions[:limit]


def save_confusion_matrix_plot(
    confusion_matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    size = max(8, min(0.55 * len(labels) + 4, 24))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(title)
    axis.set_xlabel("Predicted tag")
    axis.set_ylabel("Gold tag")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)

    if len(labels) <= 15:
        for row_index, row in enumerate(confusion_matrix):
            for col_index, value in enumerate(row):
                if value:
                    axis.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)

    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def train_and_evaluate_split(
    train_sentences: Sequence[Sequence[tuple[str, str]]],
    test_sentences: Sequence[Sequence[tuple[str, str]]],
    compare_baseline: bool = False,
    confusion_matrix_output: str | Path | None = None,
    title_prefix: str = "Brown POS Tagging",
    verbose: int = 0,
) -> dict[str, object]:
    vectorizer, classifier, default_tag = train_classifier(train_sentences, verbose=verbose)
    predicted_sentences = predict_tagged_sentences(test_sentences, classifier, vectorizer, default_tag)

    gold_tags = flatten_gold_tags(test_sentences)
    pred_tags = flatten_predicted_tags(predicted_sentences)
    metrics = compute_metrics(gold_tags, pred_tags)
    labels = metrics["labels"]
    confusion_matrix = build_confusion_matrix(gold_tags, pred_tags, labels)
    top_confusions = report_top_confusions(confusion_matrix, labels)

    saved_confusion_path: Path | None = None
    if confusion_matrix_output:
        saved_confusion_path = save_confusion_matrix_plot(
            confusion_matrix,
            labels,
            confusion_matrix_output,
            title=f"{title_prefix} confusion matrix",
        )

    results: dict[str, object] = {
        "train_sentence_count": len(train_sentences),
        "test_sentence_count": len(test_sentences),
        "predicted_sentences": predicted_sentences,
        "metrics": metrics,
        "labels": list(labels),
        "confusion_matrix": confusion_matrix,
        "top_confusions": top_confusions,
        "confusion_matrix_output": saved_confusion_path,
    }

    if compare_baseline:
        baseline_map, baseline_default_tag = build_baseline_tagger(train_sentences)
        baseline_predictions = predict_with_baseline(test_sentences, baseline_map, baseline_default_tag)
        baseline_pred_tags = flatten_predicted_tags(baseline_predictions)
        baseline_metrics = compute_metrics(gold_tags, baseline_pred_tags)
        baseline_confusion = build_confusion_matrix(gold_tags, baseline_pred_tags, baseline_metrics["labels"])
        results["baseline"] = {
            "metrics": baseline_metrics,
            "top_confusions": report_top_confusions(baseline_confusion, baseline_metrics["labels"]),
        }

    return results


def train_and_evaluate(
    tagset: str = "universal",
    test_size: float = 0.2,
    seed: int = 42,
    compare_baseline: bool = False,
    confusion_matrix_output: str | Path | None = None,
    verbose: int = 0,
) -> dict[str, object]:
    train_sentences, test_sentences = prepare_tagged_sentences(tagset=tagset, test_size=test_size, seed=seed)
    return train_and_evaluate_split(
        train_sentences,
        test_sentences,
        compare_baseline=compare_baseline,
        confusion_matrix_output=confusion_matrix_output,
        title_prefix=f"Brown POS Tagging ({tagset})",
        verbose=verbose,
    )


def format_metrics_table(metrics: dict[str, object]) -> str:
    lines = ["Tag           Precision  Recall     F1  Support", "-----------------------------------------------"]
    per_tag = metrics["per_tag"]

    for tag in metrics["labels"]:
        tag_scores = per_tag[tag]
        lines.append(
            f"{tag:<12} {tag_scores['precision']:>9.4f} "
            f"{tag_scores['recall']:>7.4f} {tag_scores['f1']:>7.4f} {tag_scores['support']:>8}"
        )

    lines.extend(
        [
            "",
            f"Macro precision: {metrics['macro_precision']:.4f}",
            f"Macro recall   : {metrics['macro_recall']:.4f}",
            f"Macro F1       : {metrics['macro_f1']:.4f}",
            f"Accuracy       : {metrics['accuracy']:.4f}",
        ]
    )
    return "\n".join(lines)


def print_results(results: dict[str, object], compare_baseline: bool) -> None:
    print(f"Train sentences: {results['train_sentence_count']:,}")
    print(f"Test sentences : {results['test_sentence_count']:,}")
    print()
    print("LinearSVC results")
    print(format_metrics_table(results["metrics"]))

    top_confusions = results["top_confusions"]
    if top_confusions:
        print()
        print("Top confusions")
        for gold_tag, pred_tag, count in top_confusions:
            print(f"- gold={gold_tag:<10} predicted={pred_tag:<10} count={count}")

    if results["confusion_matrix_output"]:
        print()
        print(f"Saved confusion matrix to: {results['confusion_matrix_output']}")

    if compare_baseline and "baseline" in results:
        print()
        print("Most-frequent-tag baseline")
        print(format_metrics_table(results["baseline"]["metrics"]))
        if results["baseline"]["top_confusions"]:
            print()
            print("Baseline top confusions")
            for gold_tag, pred_tag, count in results["baseline"]["top_confusions"]:
                print(f"- gold={gold_tag:<10} predicted={pred_tag:<10} count={count}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise for Brown corpus POS tagging.")
    parser.add_argument("--mode", choices=("train-eval",), required=True)
    parser.add_argument("--tagset", choices=VALID_TAGSETS, default="universal")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confusion-matrix-output", help="Optional path to save the confusion matrix image.")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare against a most-frequent-tag baseline.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity passed to LinearSVC/liblinear.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        results = train_and_evaluate(
            tagset=args.tagset,
            test_size=args.test_size,
            seed=args.seed,
            compare_baseline=args.compare_baseline,
            confusion_matrix_output=args.confusion_matrix_output,
            verbose=args.verbose,
        )
    except NotImplementedError as error:
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        parser.error(str(error))
        return 2

    print_results(results, compare_baseline=args.compare_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
