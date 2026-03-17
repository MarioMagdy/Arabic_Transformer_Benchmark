# Arabic Encoder Benchmark

A small benchmark project for comparing Arabic encoder models on zero-shot intent classification, plus a standalone dashboard for visualizing the results.

## What Changed

This repo now uses:

- an external benchmark dataset in [data/arabic_intent_benchmark.json](./data/arabic_intent_benchmark.json)
- a fixed global intent label space for fairer evaluation
- more Arabic test examples across sports, politics, business, greeting, shipping, returns, product recommendation, and appointment intents
- macro intent accuracy and macro category accuracy in addition to overall accuracy
- a standalone dashboard for exploring speed, confidence, margins, and failure cases

## Project Structure

```text
.
├── compare_arabic_encoders.py
├── arabic_encoder_benchmark_results.json
├── data/
│   └── arabic_intent_benchmark.json
├── index.html
├── dashboard.css
├── dashboard.js
├── plan.md
└── trying_transformers.ipynb
```

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The benchmark script uses:

- `torch`
- `transformers`

## Run The Benchmark

Run with the default external dataset:

```bash
python compare_arabic_encoders.py
```

Use a custom output path:

```bash
python compare_arabic_encoders.py --output custom_results.json
```

Disable Arabic normalization if you want raw-text comparisons:

```bash
python compare_arabic_encoders.py --no-normalize
```

Adjust thresholds or tokenizer length:

```bash
python compare_arabic_encoders.py --score-threshold 0.62 --margin-threshold 0.05 --max-length 320
```

Use a different dataset file:

```bash
python compare_arabic_encoders.py --dataset data/arabic_intent_benchmark.json
```

## Why The Benchmark Is Fairer Now

The old style of choosing label options from the expected answer makes the task easier than a real classifier.

This repo now evaluates each test against one fixed intent label space, which is more realistic because every model must choose from the full set of candidate intents:

- `sports`
- `politics`
- `business`
- `greeting`
- `shipping`
- `returns`
- `product_recommendation`
- `appointment`

The script now reports:

- overall accuracy
- macro intent accuracy
- macro category accuracy
- confident accuracy
- coverage above threshold
- average score
- average margin
- average latency

## Dataset Format

The benchmark dataset lives in [data/arabic_intent_benchmark.json](./data/arabic_intent_benchmark.json).

High-level format:

```json
{
  "metadata": {
    "name": "Arabic Intent Benchmark",
    "version": "1.1.0"
  },
  "label_sets": {
    "default": [
      { "key": "sports", "label": "..." }
    ]
  },
  "tests": [
    {
      "id": "S1",
      "text": "هذه مباراة تنس طاولة رائعة جدًا",
      "expected": "sports",
      "category": "sports",
      "label_set": "default"
    }
  ]
}
```

To add more benchmark cases:

1. Open the dataset file.
2. Add a new test object with a unique `id`.
3. Make sure `expected` exists in the selected `label_set`.
4. Re-run the benchmark script.

## Open The Dashboard

After generating `arabic_encoder_benchmark_results.json`, start a simple local server in this folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/index.html
```

The dashboard shows:

- winner and KPI overview
- sortable leaderboard
- speed vs accuracy scatter
- radar fingerprint
- category heatmap
- model cards
- sentence explorer
- failure analysis

If you open the page directly without a server, use the **Load Benchmark JSON** button and choose the results file manually.

## Notes

- The benchmark can take a while on CPU because it loads several encoder models.
- Model downloads may require internet access the first time.
- The dashboard is static HTML/CSS/JS and does not need a frontend build step.
