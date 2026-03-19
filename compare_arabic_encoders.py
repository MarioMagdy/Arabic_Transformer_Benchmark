"""
Multilingual Encoder Model Benchmark
====================================

Compares encoder models on a multilingual zero-shot intent benchmark.

Primary benchmark rules:
  1. The multilingual dataset is the source of truth and is never mutated at load time.
  2. Two text modes are evaluated:
     - original: exact dataset text
     - expanded: dataset text plus a language-matched suffix
  3. Ranking uses the original-text leaderboard by default.
  4. Category means intent family; difficulty is a separate analysis dimension.
  5. Confidence metrics are diagnostic only and are not used for ranking.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_DATASET_PATH = Path("data/multilingual_intent_benchmark.json")
DEFAULT_OUTPUT_JSON = Path("arabic_encoder_benchmark_results.json")
DEFAULT_THRESHOLDS = {
    "score": 0.60,
    "margin": 0.04,
}
DEFAULT_MAX_LENGTH = 256
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_RUNS = 1
PRIMARY_TEXT_MODE = "original"
TEXT_MODES = ("original", "expanded")
PRIMARY_BOARD = "all_models_comparison"
SUPPORTED_DIFFICULTIES = ("easy", "normal", "hard")
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

DIFFICULTY_MAP = {
    "easy": "easy",
    "simple": "easy",
    "normal": "normal",
    "mid": "normal",
    "medium": "normal",
    "hard": "hard",
}

BOARD_DEFINITIONS = {
    "multilingual_models": {
        "label": "Multilingual Models",
        "description": "Models intended for multilingual comparison and default benchmark conclusions.",
        "families": {"multilingual"},
        "languages": None,
    },
    "all_models_comparison": {
        "label": "All Models Comparison",
        "description": "Mixed comparison board that combines multilingual and Arabic-specialized models.",
        "families": None,
        "languages": None,
    },
    "arabic_focused_comparison": {
        "label": "Arabic Focused Comparison",
        "description": "All models compared on Arabic and English only, useful for Arabic-first product decisions.",
        "families": None,
        "languages": ["ar", "en"],
    },
}


@dataclass
class BenchmarkDataset:
    path: Path
    metadata: Dict[str, Any]
    tests: List[Dict[str, Any]]
    label_sets: Dict[str, List[Dict[str, str]]]
    languages: List[str]
    categories: List[str]
    difficulties: List[str]


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    family: str
    use_e5_prefix: bool = False
    use_arabic_label_wrapper: bool = True
    pooling: str = "mean"  # "mean" | "cls"


MODELS = [
    ModelConfig(
        name="multilingual-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        family="multilingual",
        use_e5_prefix=True,
        use_arabic_label_wrapper=False,
        pooling="mean",
    ),
    ModelConfig(
        name="multilingual-e5-base",
        hf_id="intfloat/multilingual-e5-base",
        family="multilingual",
        use_e5_prefix=True,
        use_arabic_label_wrapper=False,
        pooling="mean",
    ),
    ModelConfig(
        name="AraBERTv02",
        hf_id="aubmindlab/bert-base-arabertv02",
        family="arabic_specialized",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
    ModelConfig(
        name="MARBERTv2",
        hf_id="UBC-NLP/MARBERTv2",
        family="arabic_specialized",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
    ModelConfig(
        name="MiniLM-L12-multilingual",
        hf_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        family="multilingual",
        use_e5_prefix=False,
        use_arabic_label_wrapper=True,
        pooling="mean",
    ),
]


@dataclass
class TestResult:
    test_id: str
    text_mode: str
    language: str
    category: str
    difficulty: str
    text: str
    evaluated_text: str
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
    results_by_text_mode: Dict[str, List[TestResult]] = field(default_factory=dict)
    load_time_s: float = 0.0

    def board_memberships(self) -> List[str]:
        boards = ["all_models_comparison", "arabic_focused_comparison"]
        if self.cfg.family == "multilingual":
            boards.insert(0, "multilingual_models")
        return boards

    def summary_by_text_mode(self) -> Dict[str, Dict[str, Any]]:
        return {
            text_mode: summarize_results(self.cfg, results, self.load_time_s)
            for text_mode, results in self.results_by_text_mode.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "family": self.cfg.family,
            "board_memberships": self.board_memberships(),
            "load_time_s": round(float(self.load_time_s), 6),
            "summaries_by_text_mode": self.summary_by_text_mode(),
            "results_by_text_mode": {
                text_mode: [result.to_dict() for result in results]
                for text_mode, results in self.results_by_text_mode.items()
            },
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
        description="Benchmark encoder models on a multilingual intent dataset."
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


def normalize_difficulty(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key not in DIFFICULTY_MAP:
        raise ValueError(
            f"Unsupported difficulty '{value}'. Expected one of: {sorted(set(DIFFICULTY_MAP))}."
        )
    return DIFFICULTY_MAP[key]


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    metadata = raw.get("metadata", {})
    label_sets = raw.get("label_sets", {})
    tests = normalize_tests(raw.get("tests", []), label_sets)
    validate_benchmark_dataset(tests, label_sets, metadata, path)

    languages = sorted({test["language"] for test in tests})
    categories = sorted({test["category"] for test in tests})
    difficulties = [
        difficulty
        for difficulty in SUPPORTED_DIFFICULTIES
        if difficulty in {test["difficulty"] for test in tests}
    ]

    return BenchmarkDataset(
        path=path,
        metadata=metadata,
        tests=tests,
        label_sets=label_sets,
        languages=languages,
        categories=categories,
        difficulties=difficulties,
    )

def normalize_tests(
    tests: List[Dict[str, Any]],
    label_sets: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    del label_sets
    normalized_tests: List[Dict[str, Any]] = []
    for raw_test in tests:
        test = dict(raw_test)
        label_set_name = test.get("label_set", "default")
        expected = str(test.get("expected", "")).strip()
        category = str(test.get("category", "")).strip() or expected
        if category in {"faq", "hard"} and expected:
            category = expected

        difficulty_raw = test.get("difficulty")
        if difficulty_raw is None:
            inferred = infer_legacy_difficulty(test)
            difficulty = inferred if inferred is not None else ""
        else:
            difficulty = normalize_difficulty(difficulty_raw)

        normalized_tests.append(
            {
                **test,
                "id": str(test.get("id", "")).strip(),
                "text": str(test.get("text", "")).strip(),
                "expected": expected,
                "category": category,
                "difficulty": difficulty,
                "language": str(test.get("language", "")).strip(),
                "label_set": label_set_name,
            }
        )
    return normalized_tests


def infer_legacy_difficulty(test: Dict[str, Any]) -> str | None:
    test_id = str(test.get("id", "")).strip()
    match = re.search(r"_(\d+)$", test_id)
    if not match:
        match = re.search(r"[A-Za-z]+(\d+)$", test_id)
    if not match:
        return None

    index = int(match.group(1))
    if index in {1, 3}:
        return "easy"
    if index in {2, 4}:
        return "normal"
    if index >= 5:
        return "hard"
    return None


def validate_benchmark_dataset(
    tests: List[Dict[str, Any]],
    label_sets: Dict[str, List[Dict[str, str]]],
    metadata: Dict[str, Any],
    path: Path,
) -> None:
    if not tests:
        raise ValueError("Dataset must contain at least one test.")
    if not label_sets:
        raise ValueError("Dataset must contain at least one label set.")

    test_ids = [test.get("id") for test in tests]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Test IDs must be unique.")

    if any(not test.get("language") for test in tests):
        raise ValueError(
            f"Dataset '{path}' is missing the required 'language' field on one or more tests. "
            "The Arabic-only dataset is now treated as legacy input and is not supported by this script."
        )

    allowed_keys = set()
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
        allowed_keys.update(keys)

    if metadata.get("languages"):
        metadata_languages = set(metadata["languages"])
        test_languages = {test["language"] for test in tests}
        missing_from_metadata = test_languages - metadata_languages
        if missing_from_metadata:
            raise ValueError(
                f"Dataset metadata.languages is missing languages present in tests: {sorted(missing_from_metadata)}"
            )

    for test in tests:
        label_set_name = test.get("label_set", "default")
        if label_set_name not in label_sets:
            raise ValueError(
                f"Test '{test.get('id')}' references missing label set '{label_set_name}'."
            )
        for field in ("id", "text", "expected", "category", "difficulty", "language"):
            if not test.get(field):
                raise ValueError(f"Test is missing required field '{field}': {test}")

        expected = test["expected"]
        if expected not in {entry["key"] for entry in label_sets[label_set_name]}:
            raise ValueError(
                f"Test '{test['id']}' expected key '{expected}' is not present in '{label_set_name}'."
            )
        if test["category"] not in allowed_keys:
            raise ValueError(
                f"Test '{test['id']}' category '{test['category']}' is not a valid intent family label key."
            )
        if test["difficulty"] not in SUPPORTED_DIFFICULTIES:
            raise ValueError(
                f"Test '{test['id']}' difficulty '{test['difficulty']}' is invalid. "
                f"Expected one of {SUPPORTED_DIFFICULTIES}."
            )


def summarize_dataset(dataset: BenchmarkDataset) -> Dict[str, Any]:
    category_counts = Counter(test["category"] for test in dataset.tests)
    intent_counts = Counter(test["expected"] for test in dataset.tests)
    language_counts = Counter(test["language"] for test in dataset.tests)
    difficulty_counts = Counter(test["difficulty"] for test in dataset.tests)
    return {
        "num_tests": len(dataset.tests),
        "num_label_sets": len(dataset.label_sets),
        "counts_by_category": dict(sorted(category_counts.items())),
        "counts_by_expected_key": dict(sorted(intent_counts.items())),
        "counts_by_language": dict(sorted(language_counts.items())),
        "counts_by_difficulty": dict(sorted(difficulty_counts.items())),
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


def get_evaluation_text(text: str, language: str, text_mode: str) -> str:
    base_text = str(text).strip()
    if text_mode == "original":
        return base_text
    if text_mode != "expanded":
        raise ValueError(f"Unsupported text mode '{text_mode}'.")

    suffix = LONG_TEXT_SUFFIXES.get(language, "")
    if suffix and suffix not in base_text:
        return f"{base_text} {suffix}".strip()
    return base_text


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
    batch = {key: value.to(DEVICE) for key, value in batch.items()}

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


def accuracy_pct(results: List[TestResult]) -> float:
    if not results:
        return 0.0
    return sum(result.correct for result in results) / len(results) * 100.0


def confident_accuracy(results: List[TestResult]) -> Tuple[float, float]:
    if not results:
        return 0.0, 0.0
    confident_results = [result for result in results if result.confident]
    if not confident_results:
        return 0.0, 0.0
    accuracy = sum(result.correct for result in confident_results) / len(confident_results) * 100.0
    coverage = len(confident_results) / len(results) * 100.0
    return accuracy, coverage


def average_attr(results: List[TestResult], attr: str) -> float:
    if not results:
        return 0.0
    return sum(float(getattr(result, attr)) for result in results) / len(results)


def accuracy_by_field(
    results: List[TestResult],
    field: str,
    ordered_values: Iterable[str] | None = None,
) -> Dict[str, float | None]:
    output: Dict[str, float | None] = {}
    if ordered_values is None:
        values = sorted({str(getattr(result, field)) for result in results})
    else:
        values = list(ordered_values)
    for value in values:
        subset = [result for result in results if getattr(result, field) == value]
        output[value] = accuracy_pct(subset) if subset else None
    return output


def macro_accuracy(values: Dict[str, float | None]) -> float:
    filtered = [value for value in values.values() if value is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def summarize_results(
    cfg: ModelConfig,
    results: List[TestResult],
    load_time_s: float,
) -> Dict[str, Any]:
    by_category = accuracy_by_field(results, "category")
    by_expected = accuracy_by_field(results, "expected_key")
    by_language = accuracy_by_field(results, "language")
    by_difficulty = accuracy_by_field(results, "difficulty", SUPPORTED_DIFFICULTIES)
    conf_acc, coverage = confident_accuracy(results)
    text_mode = results[0].text_mode if results else PRIMARY_TEXT_MODE

    return {
        "model_name": cfg.name,
        "hf_id": cfg.hf_id,
        "family": cfg.family,
        "text_mode": text_mode,
        "load_time_s": round(float(load_time_s), 6),
        "num_results": len(results),
        "overall_accuracy_pct": round(float(accuracy_pct(results)), 6),
        "macro_intent_accuracy_pct": round(float(macro_accuracy(by_expected)), 6),
        "macro_category_accuracy_pct": round(float(macro_accuracy(by_category)), 6),
        "macro_language_accuracy_pct": round(float(macro_accuracy(by_language)), 6),
        "macro_difficulty_accuracy_pct": round(float(macro_accuracy(by_difficulty)), 6),
        "confident_accuracy_pct": round(float(conf_acc), 6),
        "coverage_above_threshold_pct": round(float(coverage), 6),
        "avg_score": round(float(average_attr(results, "score")), 6),
        "avg_margin": round(float(average_attr(results, "margin")), 6),
        "avg_latency_ms": round(float(average_attr(results, "latency_ms")), 6),
        "per_category_accuracy_pct": {
            key: (round(float(value), 6) if value is not None else None)
            for key, value in by_category.items()
        },
        "per_expected_accuracy_pct": {
            key: (round(float(value), 6) if value is not None else None)
            for key, value in by_expected.items()
        },
        "per_language_accuracy_pct": {
            key: (round(float(value), 6) if value is not None else None)
            for key, value in by_language.items()
        },
        "per_difficulty_accuracy_pct": {
            key: (round(float(value), 6) if value is not None else None)
            for key, value in by_difficulty.items()
        },
    }


def get_models_for_board(
    all_model_results: List[ModelResult],
    board_name: str,
) -> List[ModelResult]:
    board_config = BOARD_DEFINITIONS[board_name]
    families = board_config.get("families")
    return [
        model_result
        for model_result in all_model_results
        if board_name in model_result.board_memberships()
        and (families is None or model_result.cfg.family in families)
    ]


def get_board_results(
    model_result: ModelResult,
    board_name: str,
    text_mode: str,
) -> List[TestResult]:
    results = model_result.results_by_text_mode[text_mode]
    board_languages = BOARD_DEFINITIONS[board_name].get("languages")
    if not board_languages:
        return results
    return [result for result in results if result.language in set(board_languages)]


def rank_models_for_text_mode(
    model_results: List[ModelResult],
    text_mode: str,
    board_name: str,
) -> List[ModelResult]:
    return sorted(
        model_results,
        key=lambda model_result: (
            summarize_results(model_result.cfg, get_board_results(model_result, board_name, text_mode), model_result.load_time_s)[
                "macro_language_accuracy_pct"
            ],
            summarize_results(model_result.cfg, get_board_results(model_result, board_name, text_mode), model_result.load_time_s)[
                "macro_intent_accuracy_pct"
            ],
            summarize_results(model_result.cfg, get_board_results(model_result, board_name, text_mode), model_result.load_time_s)[
                "overall_accuracy_pct"
            ],
        ),
        reverse=True,
    )


def build_ranking_json(
    ranked: List[ModelResult],
    text_mode: str,
    board_name: str,
) -> List[Dict[str, Any]]:
    ranking_json = []
    for rank, model_result in enumerate(ranked, 1):
        summary = summarize_results(
            model_result.cfg,
            get_board_results(model_result, board_name, text_mode),
            model_result.load_time_s,
        )
        ranking_json.append(
            {
                "rank": rank,
                "board": board_name,
                "text_mode": text_mode,
                **summary,
            }
        )
    return ranking_json


def build_board_payloads(all_model_results: List[ModelResult]) -> Dict[str, Any]:
    boards_payload: Dict[str, Any] = {}
    for board_name, board_info in BOARD_DEFINITIONS.items():
        board_models = get_models_for_board(all_model_results, board_name)
        rankings_by_text_mode = {
            text_mode: build_ranking_json(
                rank_models_for_text_mode(board_models, text_mode, board_name),
                text_mode,
                board_name,
            )
            for text_mode in TEXT_MODES
        }
        boards_payload[board_name] = {
            "key": board_name,
            "label": board_info["label"],
            "description": board_info["description"],
            "default_languages": board_info.get("languages"),
            "model_names": [model_result.cfg.name for model_result in board_models],
            "rankings_by_text_mode": rankings_by_text_mode,
            "default_ranking": rankings_by_text_mode[PRIMARY_TEXT_MODE],
        }
    return boards_payload

def print_dataset_overview(dataset: BenchmarkDataset) -> None:
    summary = summarize_dataset(dataset)
    print("=" * 72)
    print("  DATASET")
    print("=" * 72)
    print(f"  Name:         {dataset.metadata.get('name', 'Multilingual Intent Benchmark')}")
    print(f"  Version:      {dataset.metadata.get('version', 'n/a')}")
    print(f"  Path:         {dataset.path.resolve()}")
    print(f"  Tests:        {summary['num_tests']}")
    print(f"  Intents:      {len(summary['counts_by_expected_key'])}")
    print(f"  Label sets:   {summary['num_label_sets']}")
    print(f"  Languages:    {summary['counts_by_language']}")
    print(f"  Categories:   {summary['counts_by_category']}")
    print(f"  Difficulties: {summary['counts_by_difficulty']}")
    print()


def print_summary(
    all_model_results: List[ModelResult],
    dataset: BenchmarkDataset,
) -> Dict[str, Any]:
    del dataset
    board_payloads = build_board_payloads(all_model_results)
    primary_ranking = board_payloads[PRIMARY_BOARD]["default_ranking"]

    print("\n\n" + "=" * 98)
    print("  BENCHMARK SUMMARY")
    print("=" * 98)
    print(f"  Primary board: {PRIMARY_BOARD}")
    print(f"  Primary text mode: {PRIMARY_TEXT_MODE}")
    print(f"  Secondary text mode: expanded")
    print("-" * 98)

    ranked_models = rank_models_for_text_mode(
        get_models_for_board(all_model_results, PRIMARY_BOARD),
        PRIMARY_TEXT_MODE,
        PRIMARY_BOARD,
    )
    print(row("Model", *[model_result.cfg.name for model_result in ranked_models]))
    print("-" * 98)
    print(
        row(
            "Macro Language Acc (%)",
            *[
                f"{summarize_results(model_result.cfg, model_result.results_by_text_mode[PRIMARY_TEXT_MODE], model_result.load_time_s)['macro_language_accuracy_pct']:.1f}"
                for model_result in ranked_models
            ],
        )
    )
    print(
        row(
            "Macro Intent Acc (%)",
            *[
                f"{summarize_results(model_result.cfg, model_result.results_by_text_mode[PRIMARY_TEXT_MODE], model_result.load_time_s)['macro_intent_accuracy_pct']:.1f}"
                for model_result in ranked_models
            ],
        )
    )
    print(
        row(
            "Overall Accuracy (%)",
            *[
                f"{summarize_results(model_result.cfg, model_result.results_by_text_mode[PRIMARY_TEXT_MODE], model_result.load_time_s)['overall_accuracy_pct']:.1f}"
                for model_result in ranked_models
            ],
        )
    )
    print(
        row(
            "Avg Latency (ms)",
            *[
                f"{summarize_results(model_result.cfg, model_result.results_by_text_mode[PRIMARY_TEXT_MODE], model_result.load_time_s)['avg_latency_ms']:.1f}"
                for model_result in ranked_models
            ],
        )
    )
    print("-" * 98)
    print(f"\n  PRIMARY LEADERBOARD (original text, {PRIMARY_BOARD})")
    print("  " + "-" * 82)
    for entry in primary_ranking:
        print(
            f"  #{entry['rank']}  {entry['model_name']:<30}  "
            f"macro_lang={entry['macro_language_accuracy_pct']:5.1f}%  "
            f"macro_intent={entry['macro_intent_accuracy_pct']:5.1f}%  "
            f"overall={entry['overall_accuracy_pct']:5.1f}%  "
            f"lat={entry['avg_latency_ms']:.0f}ms"
        )

    print("\n  PER-LANGUAGE ACCURACY BY MODEL (original text)")
    print("  " + "-" * 82)
    for model_result in ranked_models:
        summary = summarize_results(
            model_result.cfg,
            model_result.results_by_text_mode[PRIMARY_TEXT_MODE],
            model_result.load_time_s,
        )
        per_language = summary["per_language_accuracy_pct"]
        language_bits = ", ".join(
            f"{language}={per_language.get(language, 0.0):.1f}%"
            for language in sorted(per_language)
        )
        print(f"  {model_result.cfg.name:<30}  {language_bits}")

    return {
        "boards": board_payloads,
        "ranking": primary_ranking,
    }


def export_results(
    all_model_results: List[ModelResult],
    board_payloads: Dict[str, Any],
    default_ranking: List[Dict[str, Any]],
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
            "num_models": len(all_model_results),
            "num_label_sets": len(dataset.label_sets),
            "thresholds": thresholds,
            "max_length": max_length,
            "normalization_enabled": normalize_inputs,
            "repeats": repeats,
            "warmup_runs": warmup_runs,
            "text_modes": list(TEXT_MODES),
            "primary_text_mode": PRIMARY_TEXT_MODE,
            "primary_board": PRIMARY_BOARD,
            "default_dashboard_view": {
                "board": PRIMARY_BOARD,
                "text_mode": PRIMARY_TEXT_MODE,
                "languages": BOARD_DEFINITIONS[PRIMARY_BOARD].get("languages") or dataset.languages,
                "difficulties": dataset.difficulties,
            },
            "counts_by_language": dataset_summary["counts_by_language"],
            "counts_by_category": dataset_summary["counts_by_category"],
            "counts_by_difficulty": dataset_summary["counts_by_difficulty"],
            "counts_by_expected_key": dataset_summary["counts_by_expected_key"],
            "generated_at_unix": time.time(),
        },
        "dataset_metadata": dataset.metadata,
        "tests": dataset.tests,
        "label_sets": dataset.label_sets,
        "boards": board_payloads,
        "models": [model_result.to_dict() for model_result in all_model_results],
        "ranking": default_ranking,
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
    print("=" * 72)
    print("  LOADING MODELS & RUNNING BENCHMARK")
    print("=" * 72)

    for cfg in MODELS:
        print(f"\n{'-' * 72}")
        print(f"  MODEL: {cfg.name}  [{cfg.family}]")
        print(f"{'-' * 72}")

        start_load = time.perf_counter()
        tokenizer, model = load_model(cfg)
        load_time = time.perf_counter() - start_load

        model_result = ModelResult(
            cfg=cfg,
            load_time_s=load_time,
            results_by_text_mode={text_mode: [] for text_mode in TEXT_MODES},
        )
        label_cache: Dict[Tuple[str, ...], Any] = {}

        try:
            for text_mode in TEXT_MODES:
                print(f"  Text mode: {text_mode}")
                for test in dataset.tests:
                    labels, correct_idx = get_labels_and_expected_idx(test, dataset.label_sets)
                    evaluated_text = get_evaluation_text(test["text"], test["language"], text_mode)

                    (
                        pred_idx,
                        score,
                        margin,
                        sims,
                        latency_ms,
                        latency_std_ms,
                        latency_runs_ms,
                    ) = run_repeated_classification(
                        evaluated_text,
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
                        text_mode=text_mode,
                        language=test["language"],
                        category=test["category"],
                        difficulty=test["difficulty"],
                        text=test["text"],
                        evaluated_text=evaluated_text,
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
                    model_result.results_by_text_mode[text_mode].append(result)

                    tick = "C" if correct else "W"
                    conf = "H" if confident else "L"
                    print(
                        f"    [{tick}|{conf}]  {text_mode:<8}  {test['id']:<18}  "
                        f"score={score:.4f}  margin={margin:.4f}  "
                        f"lat={latency_ms:6.1f}ms  "
                        f"| {test['text'][:36]}"
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

    repeats = max(1, args.repeats)
    warmup_runs = max(0, args.warmup_runs)
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
    print(f"Text modes: {TEXT_MODES}")
    print(f"Primary board/text mode: {PRIMARY_BOARD} / {PRIMARY_TEXT_MODE}")
    print(f"Repeats per test: {repeats}")
    print(f"Warmup runs per test: {warmup_runs}\n")

    all_model_results = run_benchmark(
        dataset,
        thresholds,
        max_length=args.max_length,
        normalize_inputs=normalize_inputs,
        repeats=repeats,
        warmup_runs=warmup_runs,
    )
    summary_payload = print_summary(all_model_results, dataset)
    export_results(
        all_model_results,
        summary_payload["boards"],
        summary_payload["ranking"],
        dataset,
        args.output,
        thresholds,
        normalize_inputs=normalize_inputs,
        max_length=args.max_length,
        repeats=repeats,
        warmup_runs=warmup_runs,
    )


if __name__ == "__main__":
    main()
