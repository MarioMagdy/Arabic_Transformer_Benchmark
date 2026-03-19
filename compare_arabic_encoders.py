"""
Multilingual Encoder Model Benchmark
================================
Compares 5 encoder models on zero-shot multilingual intent classification.

Key improvements over the original script:
  1. Loads tests from an external JSON dataset file
  2. Uses a fixed global intent label space for fairer evaluation
  3. Supports Arabic text normalization for more stable comparisons
  4. Reports macro intent/category accuracy in addition to overall accuracy
  5. Exposes CLI options for dataset path, output path, thresholds, and max length
"""

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DATASET_PATH = Path("data/multilingual_intent_benchmark.json")
DEFAULT_OUTPUT_JSON = Path("arabic_encoder_benchmark_results.json")
DEFAULT_THRESHOLDS = {
    "score": 0.60,
    "margin": 0.04,
}
DEFAULT_MAX_LENGTH = 256
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_RUNS = 1
DEVICE = "cpu"
COL = 26

torch = None
F = None
AutoModel = None
AutoTokenizer = None

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
MULTISPACE_RE = re.compile(r"\s+")
ARABIC_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
    }
)

LONG_TEXT_SUFFIXES = {
    "ar": "ويتضمن النص أيضا تفاصيل إضافية وسياقا أوضح حول الفكرة الأساسية المقصودة.",
    "en": "It also includes extra details and clearer context about the main intended idea.",
    "fr": "Il contient aussi des détails supplémentaires et un contexte plus clair sur l'idée principale visée.",
    "es": "También incluye detalles adicionales y un contexto más claro sobre la idea principal prevista.",
}


@dataclass
class BenchmarkDataset:
    path: Path
    metadata: Dict[str, Any]
    tests: List[Dict[str, Any]]
    label_sets: Dict[str, List[Dict[str, str]]]


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    use_e5_prefix: bool = False
    use_arabic_label_wrapper: bool = True
    pooling: str = "mean"  # "mean" | "cls"


MODELS = [
    ModelConfig(
        name="multilingual-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        use_e5_prefix=True,
        use_arabic_label_wrapper=False,
        pooling="mean",
    ),
    ModelConfig(
        name="multilingual-e5-base",
        hf_id="intfloat/multilingual-e5-base",
        use_e5_prefix=True,
        use_arabic_label_wrapper=False,
        pooling="mean",
    ),
    ModelConfig(
        name="AraBERTv02",
        hf_id="aubmindlab/bert-base-arabertv02",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
    ModelConfig(
        name="MARBERTv2",
        hf_id="UBC-NLP/MARBERTv2",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
    ModelConfig(
        name="MiniLM-L12-multilingual",
        hf_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
]


@dataclass
class TestResult:
    test_id: str
    language: str
    category: str
    text: str
    expected_key: str
    correct_idx: int
    correct_label: str
    pred_idx: int
    pred_label: str
    score: float
    margin: float
    all_scores: List[float]
    labels: List[str]
    correct: bool
    confident: bool
    latency_ms: float
    latency_std_ms: float
    latency_runs_ms: List[float]
    repeat_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult:
    cfg: ModelConfig
    results: List[TestResult] = field(default_factory=list)
    load_time_s: float = 0.0

    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results) * 100.0

    def confident_accuracy(self) -> Tuple[float, float]:
        confident_results = [r for r in self.results if r.confident]
        if not confident_results:
            return 0.0, 0.0
        acc = sum(r.correct for r in confident_results) / len(confident_results) * 100.0
        cov = len(confident_results) / len(self.results) * 100.0
        return acc, cov

    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def avg_margin(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.margin for r in self.results) / len(self.results)

    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def per_category_accuracy(self) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        categories = sorted(set(r.category for r in self.results))
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            out[cat] = (
                sum(r.correct for r in cat_results) / len(cat_results) * 100.0
                if cat_results
                else None
            )
        return out

    def per_expected_accuracy(self) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        expected_keys = sorted(set(r.expected_key for r in self.results))
        for key in expected_keys:
            key_results = [r for r in self.results if r.expected_key == key]
            out[key] = (
                sum(r.correct for r in key_results) / len(key_results) * 100.0
                if key_results
                else None
            )
        return out

    def macro_category_accuracy(self) -> float:
        values = [v for v in self.per_category_accuracy().values() if v is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def macro_expected_accuracy(self) -> float:
        values = [v for v in self.per_expected_accuracy().values() if v is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def per_language_accuracy(self) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        languages = sorted(set(r.language for r in self.results))
        for language in languages:
            language_results = [r for r in self.results if r.language == language]
            out[language] = (
                sum(r.correct for r in language_results) / len(language_results) * 100.0
                if language_results
                else None
            )
        return out

    def macro_language_accuracy(self) -> float:
        values = [v for v in self.per_language_accuracy().values() if v is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def summary_dict(self) -> Dict[str, Any]:
        conf_acc, cov = self.confident_accuracy()
        return {
            "model_name": self.cfg.name,
            "hf_id": self.cfg.hf_id,
            "load_time_s": round(float(self.load_time_s), 6),
            "overall_accuracy_pct": round(float(self.accuracy()), 6),
            "macro_intent_accuracy_pct": round(float(self.macro_expected_accuracy()), 6),
            "macro_category_accuracy_pct": round(float(self.macro_category_accuracy()), 6),
            "macro_language_accuracy_pct": round(float(self.macro_language_accuracy()), 6),
            "confident_accuracy_pct": round(float(conf_acc), 6),
            "coverage_above_threshold_pct": round(float(cov), 6),
            "avg_score": round(float(self.avg_score()), 6),
            "avg_margin": round(float(self.avg_margin()), 6),
            "avg_latency_ms": round(float(self.avg_latency()), 6),
            "per_category_accuracy_pct": {
                k: (round(float(v), 6) if v is not None else None)
                for k, v in self.per_category_accuracy().items()
            },
            "per_expected_accuracy_pct": {
                k: (round(float(v), 6) if v is not None else None)
                for k, v in self.per_expected_accuracy().items()
            },
            "per_language_accuracy_pct": {
                k: (round(float(v), 6) if v is not None else None)
                for k, v in self.per_language_accuracy().items()
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "load_time_s": round(float(self.load_time_s), 6),
            "summary": self.summary_dict(),
            "results": [r.to_dict() for r in self.results],
        }


def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


def initialize_runtime() -> None:
    global torch, F, AutoModel, AutoTokenizer, DEVICE

    import torch as torch_lib
    import torch.nn.functional as functional_lib
    from transformers import AutoModel as auto_model_cls
    from transformers import AutoTokenizer as auto_tokenizer_cls

    torch = torch_lib
    F = functional_lib
    AutoModel = auto_model_cls
    AutoTokenizer = auto_tokenizer_cls
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark encoder models on an external multilingual intent dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to the JSON output file.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["score"],
        help="Minimum cosine score to mark a prediction as confident.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["margin"],
        help="Minimum top-1 vs top-2 margin to mark a prediction as confident.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max_length for queries and labels.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable Arabic normalization before encoding.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of timed inference repeats per test for averaging.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Number of untimed warmup runs before repeated timing.",
    )
    return parser.parse_args()


def normalize_arabic_text(text: str) -> str:
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(ARABIC_CHAR_MAP)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_text(text: str, normalize_inputs: bool) -> str:
    text = text.strip()
    return normalize_arabic_text(text) if normalize_inputs else text


def expand_test_texts(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded_tests: List[Dict[str, Any]] = []
    for test in tests:
        suffix = LONG_TEXT_SUFFIXES.get(test.get("language", ""))
        text = str(test.get("text", "")).strip()
        if suffix and suffix not in text:
            text = f"{text} {suffix}".strip()
        expanded_tests.append({**test, "text": text})
    return expanded_tests


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    metadata = raw.get("metadata", {})
    tests = expand_test_texts(raw.get("tests", []))
    label_sets = raw.get("label_sets", {})

    validate_benchmark_dataset(tests, label_sets)
    return BenchmarkDataset(path=path, metadata=metadata, tests=tests, label_sets=label_sets)


def validate_benchmark_dataset(
    tests: List[Dict[str, Any]],
    label_sets: Dict[str, List[Dict[str, str]]],
) -> None:
    if not tests:
        raise ValueError("Dataset must contain at least one test.")
    if not label_sets:
        raise ValueError("Dataset must contain at least one label set.")

    test_ids = [test.get("id") for test in tests]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Test IDs must be unique.")

    for set_name, entries in label_sets.items():
        if not entries:
            raise ValueError(f"Label set '{set_name}' is empty.")
        keys = [entry.get("key") for entry in entries]
        labels = [entry.get("label") for entry in entries]
        if any(not key for key in keys):
            raise ValueError(f"Label set '{set_name}' contains an empty key.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Label set '{set_name}' contains duplicate keys.")
        if any(not label for label in labels):
            raise ValueError(f"Label set '{set_name}' contains an empty label description.")

    for test in tests:
        label_set_name = test.get("label_set", "default")
        if label_set_name not in label_sets:
            raise ValueError(
                f"Test '{test.get('id')}' references missing label set '{label_set_name}'."
            )
        expected = test.get("expected")
        if expected not in {entry["key"] for entry in label_sets[label_set_name]}:
            raise ValueError(
                f"Test '{test.get('id')}' expected key '{expected}' is not present in '{label_set_name}'."
            )
        for field in ("id", "text", "expected", "category", "language"):
            if not test.get(field):
                raise ValueError(f"Test is missing required field '{field}': {test}")


def summarize_dataset(dataset: BenchmarkDataset) -> Dict[str, Any]:
    category_counts = Counter(test["category"] for test in dataset.tests)
    intent_counts = Counter(test["expected"] for test in dataset.tests)
    language_counts = Counter(test["language"] for test in dataset.tests)
    return {
        "num_tests": len(dataset.tests),
        "num_label_sets": len(dataset.label_sets),
        "counts_by_category": dict(sorted(category_counts.items())),
        "counts_by_expected_key": dict(sorted(intent_counts.items())),
        "counts_by_language": dict(sorted(language_counts.items())),
    }


def get_labels_and_expected_idx(
    test: Dict[str, Any],
    label_sets: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[str], int]:
    label_set_name = test.get("label_set", "default")
    entries = label_sets[label_set_name]
    labels = [entry["label"] for entry in entries]
    expected_key = test["expected"]
    correct_idx = next(index for index, entry in enumerate(entries) if entry["key"] == expected_key)
    return labels, correct_idx


def load_model(cfg: ModelConfig):
    print(f"  Loading {cfg.name} ({cfg.hf_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    model = AutoModel.from_pretrained(cfg.hf_id).to(DEVICE)
    model.eval()
    return tokenizer, model


def build_query_and_labels(
    text: str,
    labels: List[str],
    cfg: ModelConfig,
    normalize_inputs: bool,
    *,
    language: str,
) -> Tuple[str, List[str]]:
    clean_text = preprocess_text(text, normalize_inputs)
    clean_labels = [preprocess_text(label, normalize_inputs) for label in labels]

    if cfg.use_e5_prefix:
        return f"query: {clean_text}", [f"passage: {label}" for label in clean_labels]
    if cfg.use_arabic_label_wrapper and language != "ar":
        return clean_text, clean_labels
    if cfg.use_arabic_label_wrapper:
        return clean_text, [f"هذا النص يتحدث عن {label}" for label in clean_labels]
    return clean_text, clean_labels


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    pooling: str,
    max_length: int,
) -> Any:
    batch = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    if pooling == "cls":
        emb = out.last_hidden_state[:, 0, :]
    else:
        mask = batch["attention_mask"].unsqueeze(-1).bool()
        hidden = out.last_hidden_state.masked_fill(~mask, 0.0)
        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1).unsqueeze(-1)
        emb = hidden.sum(dim=1) / lengths

    return F.normalize(emb, p=2, dim=1)


def get_label_embeddings(
    labels: List[str],
    tokenizer,
    model,
    cfg: ModelConfig,
    cache: Dict[Tuple[str, ...], Any],
    *,
    language: str,
    normalize_inputs: bool,
    max_length: int,
) -> Any:
    _, label_texts = build_query_and_labels(
        "",
        labels,
        cfg,
        normalize_inputs,
        language=language,
    )
    label_key = tuple(label_texts)
    if label_key not in cache:
        cache[label_key] = encode_texts(label_texts, tokenizer, model, cfg.pooling, max_length)
    return cache[label_key]


def classify(
    text: str,
    labels: List[str],
    tokenizer,
    model,
    cfg: ModelConfig,
    label_cache: Dict[Tuple[str, ...], Any],
    *,
    language: str,
    normalize_inputs: bool,
    max_length: int,
):
    q_text, _ = build_query_and_labels(
        text,
        labels,
        cfg,
        normalize_inputs,
        language=language,
    )
    q_emb = encode_texts([q_text], tokenizer, model, cfg.pooling, max_length)
    l_emb = get_label_embeddings(
        labels,
        tokenizer,
        model,
        cfg,
        label_cache,
        language=language,
        normalize_inputs=normalize_inputs,
        max_length=max_length,
    )
    sims = (q_emb @ l_emb.T).squeeze(0)

    sorted_scores, _ = torch.sort(sims, descending=True)
    best_idx = int(torch.argmax(sims).item())
    best_score = float(sims[best_idx].item())
    margin = float((sorted_scores[0] - sorted_scores[1]).item()) if len(labels) > 1 else 0.0
    return best_idx, best_score, margin, sims


def run_repeated_classification(
    text: str,
    labels: List[str],
    tokenizer,
    model,
    cfg: ModelConfig,
    label_cache: Dict[Tuple[str, ...], Any],
    *,
    language: str,
    normalize_inputs: bool,
    max_length: int,
    repeats: int,
    warmup_runs: int,
):
    for _ in range(max(0, warmup_runs)):
        classify(
            text,
            labels,
            tokenizer,
            model,
            cfg,
            label_cache,
            language=language,
            normalize_inputs=normalize_inputs,
            max_length=max_length,
        )

    latency_runs_ms: List[float] = []
    sims_runs = []
    for _ in range(max(1, repeats)):
        start_infer = time.perf_counter()
        _, _, _, sims = classify(
            text,
            labels,
            tokenizer,
            model,
            cfg,
            label_cache,
            language=language,
            normalize_inputs=normalize_inputs,
            max_length=max_length,
        )
        latency_runs_ms.append((time.perf_counter() - start_infer) * 1000.0)
        sims_runs.append(sims)

    mean_sims = torch.stack(sims_runs, dim=0).mean(dim=0)
    sorted_scores, _ = torch.sort(mean_sims, descending=True)
    pred_idx = int(torch.argmax(mean_sims).item())
    score = float(mean_sims[pred_idx].item())
    margin = float((sorted_scores[0] - sorted_scores[1]).item()) if len(labels) > 1 else 0.0
    avg_latency_ms = sum(latency_runs_ms) / len(latency_runs_ms)
    variance = sum((latency - avg_latency_ms) ** 2 for latency in latency_runs_ms) / len(latency_runs_ms)
    latency_std_ms = variance ** 0.5

    return pred_idx, score, margin, mean_sims, avg_latency_ms, latency_std_ms, latency_runs_ms


def row(label: str, *vals: Any) -> str:
    parts = [f"{label:<{COL}}"]
    for value in vals:
        parts.append(f"{str(value):>15}")
    return "  ".join(parts)


def print_dataset_overview(dataset: BenchmarkDataset) -> None:
    summary = summarize_dataset(dataset)
    print("=" * 65)
    print("  DATASET")
    print("=" * 65)
    print(f"  Name:    {dataset.metadata.get('name', 'Arabic Intent Benchmark')}")
    print(f"  Version: {dataset.metadata.get('version', 'n/a')}")
    print(f"  Path:    {dataset.path.resolve()}")
    print(f"  Tests:   {summary['num_tests']}")
    print(f"  Intents: {len(summary['counts_by_expected_key'])}")
    print(f"  Label sets: {summary['num_label_sets']}")
    print(f"  By language: {summary['counts_by_language']}")
    print(f"  By category: {summary['counts_by_category']}")
    print(f"  By intent:   {summary['counts_by_expected_key']}")
    print()


def print_summary(
    all_model_results: List[ModelResult],
    dataset: BenchmarkDataset,
) -> List[ModelResult]:
    print("\n\n" + "=" * 90)
    print("  BENCHMARK SUMMARY")
    print("=" * 90)
    print(row("Model", *[mr.cfg.name for mr in all_model_results]))
    print("-" * 90)
    print(row("Overall Accuracy (%)", *[f"{mr.accuracy():.1f}" for mr in all_model_results]))
    print(row("Macro Intent Acc (%)", *[f"{mr.macro_expected_accuracy():.1f}" for mr in all_model_results]))
    print(row("Macro Category Acc (%)", *[f"{mr.macro_category_accuracy():.1f}" for mr in all_model_results]))
    print(row("Macro Language Acc (%)", *[f"{mr.macro_language_accuracy():.1f}" for mr in all_model_results]))

    conf_accs = [mr.confident_accuracy() for mr in all_model_results]
    print(row("Confident Accuracy (%)", *[f"{ca:.1f}" for ca, _ in conf_accs]))
    print(row("Coverage above threshold", *[f"{cov:.1f}%" for _, cov in conf_accs]))
    print(row("Avg Cosine Score", *[f"{mr.avg_score():.4f}" for mr in all_model_results]))
    print(row("Avg Margin", *[f"{mr.avg_margin():.4f}" for mr in all_model_results]))
    print(row("Avg Latency (ms)", *[f"{mr.avg_latency():.1f}" for mr in all_model_results]))
    print(row("Load Time (s)", *[f"{mr.load_time_s:.1f}" for mr in all_model_results]))
    print("-" * 90)

    print("\n  PER-TEST BREAKDOWN (C = correct, W = wrong, H = high-conf, L = low-conf)\n")
    header = f"  {'ID':<8}  {'Lang':<6}  {'Category':<10}  {'Text':<42}"
    for mr in all_model_results:
        header += f"  {mr.cfg.name[:15]:>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, test in enumerate(dataset.tests):
        line = (
            f"  {test['id']:<8}  {test['language']:<6}  {test['category']:<10}  "
            f"{test['text'][:40]:<42}"
        )
        for mr in all_model_results:
            result = mr.results[i]
            sym = ("C" if result.correct else "W") + ("H" if result.confident else "L")
            line += f"  {f'{sym} {result.score:.3f}':>15}"
        print(line)

    categories = sorted(set(test["category"] for test in dataset.tests))
    print("\n  PER-CATEGORY ACCURACY (%)\n")
    cat_header = f"  {'Category':<12}"
    for mr in all_model_results:
        cat_header += f"  {mr.cfg.name[:15]:>15}"
    print(cat_header)
    print("  " + "-" * (len(cat_header) - 2))

    for cat in categories:
        line = f"  {cat:<12}"
        for mr in all_model_results:
            cat_results = [r for r in mr.results if r.category == cat]
            acc = sum(r.correct for r in cat_results) / len(cat_results) * 100.0 if cat_results else 0.0
            line += f"  {acc:>14.0f}%"
        print(line)

    languages = sorted(set(test["language"] for test in dataset.tests))
    print("\n  PER-LANGUAGE ACCURACY (%)\n")
    lang_header = f"  {'Language':<12}"
    for mr in all_model_results:
        lang_header += f"  {mr.cfg.name[:15]:>15}"
    print(lang_header)
    print("  " + "-" * (len(lang_header) - 2))

    for language in languages:
        line = f"  {language:<12}"
        for mr in all_model_results:
            language_results = [r for r in mr.results if r.language == language]
            acc = (
                sum(r.correct for r in language_results) / len(language_results) * 100.0
                if language_results
                else 0.0
            )
            line += f"  {acc:>14.0f}%"
        print(line)

    print("\n  RANKING BY MACRO / OVERALL ACCURACY")
    print("  " + "-" * 58)
    ranked = sorted(
        all_model_results,
        key=lambda mr: (mr.macro_expected_accuracy(), mr.accuracy(), mr.avg_margin()),
        reverse=True,
    )

    for rank, mr in enumerate(ranked, 1):
        conf_acc, cov = mr.confident_accuracy()
        print(
            f"  #{rank}  {mr.cfg.name:<30}  "
            f"macro={mr.macro_expected_accuracy():5.1f}%  "
            f"acc={mr.accuracy():5.1f}%  "
            f"conf_acc={conf_acc:5.1f}%  "
            f"coverage={cov:5.1f}%  "
            f"avg_margin={mr.avg_margin():.4f}  "
            f"lat={mr.avg_latency():.0f}ms"
        )

    return ranked


def build_ranking_json(ranked: List[ModelResult]) -> List[Dict[str, Any]]:
    ranking_json = []
    for rank, mr in enumerate(ranked, 1):
        conf_acc, cov = mr.confident_accuracy()
        ranking_json.append(
            {
                "rank": rank,
                "model_name": mr.cfg.name,
                "hf_id": mr.cfg.hf_id,
                "overall_accuracy_pct": round(float(mr.accuracy()), 6),
                "macro_intent_accuracy_pct": round(float(mr.macro_expected_accuracy()), 6),
                "macro_category_accuracy_pct": round(float(mr.macro_category_accuracy()), 6),
                "macro_language_accuracy_pct": round(float(mr.macro_language_accuracy()), 6),
                "confident_accuracy_pct": round(float(conf_acc), 6),
                "coverage_above_threshold_pct": round(float(cov), 6),
                "avg_score": round(float(mr.avg_score()), 6),
                "avg_margin": round(float(mr.avg_margin()), 6),
                "avg_latency_ms": round(float(mr.avg_latency()), 6),
                "load_time_s": round(float(mr.load_time_s), 6),
            }
        )
    return ranking_json


def export_results(
    all_model_results: List[ModelResult],
    ranked: List[ModelResult],
    dataset: BenchmarkDataset,
    output_path: Path,
    thresholds: Dict[str, float],
    *,
    normalize_inputs: bool,
    max_length: int,
    repeats: int,
    warmup_runs: int,
) -> None:
    dataset_summary = summarize_dataset(dataset)
    export_data = {
        "metadata": {
            "benchmark_name": "Multilingual Encoder Model Benchmark",
            "dataset_name": dataset.metadata.get("name"),
            "dataset_version": dataset.metadata.get("version"),
            "dataset_path": str(dataset.path.resolve()),
            "device": str(DEVICE),
            "num_tests": len(dataset.tests),
            "num_models": len(MODELS),
            "num_label_sets": len(dataset.label_sets),
            "thresholds": thresholds,
            "max_length": max_length,
            "normalization_enabled": normalize_inputs,
            "repeats": repeats,
            "warmup_runs": warmup_runs,
            "counts_by_language": dataset_summary["counts_by_language"],
            "counts_by_category": dataset_summary["counts_by_category"],
            "counts_by_expected_key": dataset_summary["counts_by_expected_key"],
            "generated_at_unix": time.time(),
        },
        "dataset_metadata": dataset.metadata,
        "tests": dataset.tests,
        "label_sets": dataset.label_sets,
        "models": [mr.to_dict() for mr in all_model_results],
        "ranking": build_ranking_json(ranked),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(export_data, file, ensure_ascii=False, indent=2)

    print(f"\n  JSON results saved to: {output_path.resolve()}")
    print("\n  Done.\n")


def run_benchmark(
    dataset: BenchmarkDataset,
    thresholds: Dict[str, float],
    *,
    max_length: int,
    normalize_inputs: bool,
    repeats: int,
    warmup_runs: int,
) -> List[ModelResult]:
    all_model_results: List[ModelResult] = []

    print(f"Device: {DEVICE}\n")
    print("=" * 65)
    print("  LOADING MODELS & RUNNING BENCHMARK")
    print("=" * 65)

    for cfg in MODELS:
        print(f"\n{'-' * 65}")
        print(f"  MODEL: {cfg.name}")
        print(f"{'-' * 65}")

        start_load = time.perf_counter()
        tokenizer, model = load_model(cfg)
        load_time = time.perf_counter() - start_load

        model_result = ModelResult(cfg=cfg, load_time_s=load_time)
        label_cache: Dict[Tuple[str, ...], Any] = {}

        try:
            for test in dataset.tests:
                labels, correct_idx = get_labels_and_expected_idx(test, dataset.label_sets)

                pred_idx, score, margin, sims, latency_ms, latency_std_ms, latency_runs_ms = run_repeated_classification(
                    test["text"],
                    labels,
                    tokenizer,
                    model,
                    cfg,
                    label_cache,
                    language=test["language"],
                    normalize_inputs=normalize_inputs,
                    max_length=max_length,
                    repeats=repeats,
                    warmup_runs=warmup_runs,
                )

                confident = score >= thresholds["score"] and margin >= thresholds["margin"]
                correct = pred_idx == correct_idx

                result = TestResult(
                    test_id=test["id"],
                    language=test["language"],
                    category=test["category"],
                    text=test["text"],
                    expected_key=test["expected"],
                    correct_idx=correct_idx,
                    correct_label=labels[correct_idx],
                    pred_idx=pred_idx,
                    pred_label=labels[pred_idx],
                    score=float(score),
                    margin=float(margin),
                    all_scores=[float(x) for x in sims.detach().cpu().tolist()],
                    labels=labels,
                    correct=bool(correct),
                    confident=bool(confident),
                    latency_ms=float(latency_ms),
                    latency_std_ms=float(latency_std_ms),
                    latency_runs_ms=[float(value) for value in latency_runs_ms],
                    repeat_count=max(1, repeats),
                )
                model_result.results.append(result)

                tick = "C" if correct else "W"
                conf = "H" if confident else "L"
                print(
                    f"  [{tick}|{conf}]  {test['id']:<6}  "
                    f"score={score:.4f}  margin={margin:.4f}  "
                    f"latency={latency_ms:6.1f}ms  "
                    f"std={latency_std_ms:5.1f}  "
                    f"| {test['text'][:40]}"
                )

            all_model_results.append(model_result)
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_model_results


def main() -> None:
    configure_stdio()
    args = parse_args()
    initialize_runtime()

    dataset = load_benchmark_dataset(args.dataset)
    thresholds = {
        "score": args.score_threshold,
        "margin": args.margin_threshold,
    }
    normalize_inputs = not args.no_normalize

    print_dataset_overview(dataset)
    print(f"Normalization enabled: {normalize_inputs}")
    print(f"Max length: {args.max_length}")
    print(f"Thresholds: {thresholds}")
    print(f"Repeats per test: {max(1, args.repeats)}")
    print(f"Warmup runs per test: {max(0, args.warmup_runs)}\n")

    all_model_results = run_benchmark(
        dataset,
        thresholds,
        max_length=args.max_length,
        normalize_inputs=normalize_inputs,
        repeats=max(1, args.repeats),
        warmup_runs=max(0, args.warmup_runs),
    )
    ranked = print_summary(all_model_results, dataset)
    export_results(
        all_model_results,
        ranked,
        dataset,
        args.output,
        thresholds,
        normalize_inputs=normalize_inputs,
        max_length=args.max_length,
        repeats=max(1, args.repeats),
        warmup_runs=max(0, args.warmup_runs),
    )


if __name__ == "__main__":
    main()
