# Facial-Operator demo (prototype)

This is a working prototype that reads webcam frames, detects facial landmarks and hands, and attempts to
recognize simple operator expressions from facial gestures while using hand-mounted digit detection to build
tiny arithmetic expressions (for example: "3 + 2 = 5"). It's intended as a demo of the full data
pipeline — collection, training, inference — rather than a finished product.

What you'll find here
- Live demo: `python main.py demo` — runs MediaPipe Face Mesh + Hands, shows a mouth preview, predicted operator (rule-based), and the running arithmetic result.
- Data collector: `python main.py collect` — press keys 1..N to label and save 224×224 mouth crops into an ImageFolder-style dataset.
- Simple, explainable baseline: this lightweight repo uses a rule-based geometric mouth heuristic for operator detection so the demo is easy to run without heavy ML dependencies.

Quick start (macOS, zsh)

1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pillow scikit-learn
```

2) Collect some data (optional but recommended)

```bash
python main.py collect --out data/collected --classes neutral,frown,smile,tongue,open
# Press 1..5 to save samples for the listed classes. Press 'v' to toggle saving to val instead of train.
```

3) Run the demo

```bash
python main.py demo --smoother 7
```

Operator labels and mapping
- The prototype uses class names (folder names) as labels when training. For the demo, numeric model outputs map to operator symbols by default as `['+', '−', 'x', '÷']`.
- When you collect real data, choose class folder names that make sense for your mapping (e.g. `smile` -> `+`, `frown` -> `−`, `tongue` -> `x`, `open` -> `÷`). We can make this mapping configurable if needed.

Issues you may see
- On macOS you may see a Continuity Camera warning about AVCaptureDeviceTypeExternal; it's harmless for running the script.
- MediaPipe / TensorFlow prints some startup logs; these are informational. If you want quieter output I can further suppress them.

Notes and recommendations
- The demo is a prototype: collect real mouth-crop data across multiple subjects and lighting conditions for best results.
- Use augmentation (flips, brightness/color jitter) and class balancing when training on small datasets.

Want it polished for a TA demo?
- I can add a short `docs/demo.gif`, a small checkpoint for quick verification, and a brief Troubleshooting section. Ask and I will add them.

Files added in this prototype
- `requirements.txt` — Python dependencies
- `main.py` — simple entrypoint to launch demo or collector
- `src/` — demo, model, utils, dataset, training and evaluation scripts
- `scripts/` — synthetic-data generator and training stub

If anything is unclear or you'd like the README to present different commands or wording, tell me which tone or phrasing you prefer and I'll update it.
