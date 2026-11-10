"""Benchmark + sanity-check the NLI models outside the app.
Run from project root:  python bench_nli.py
"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(os.cpu_count() or 4)
print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | CPU threads: {torch.get_num_threads()}\n")

PREMISE = ("In principle, those tasks can in principle be formulated as a POMDP which is defined as a tuple of six "
           "components {S, A, P, r, Ω, Z}, where S, A, P, r are the state space, action space, transition function "
           "and reward function, respectively as in a Markov decision process. Ω and Z are observation space and "
           "observation model, respectively.")
CASES = [
    ("verbatim copy (expect ENTAILMENT)", "Ω and Z are observation space and observation model, respectively."),
    ("contradiction (expect CONTRADICTION)", "A POMDP is defined as a tuple of three components."),
    ("unrelated (expect NEUTRAL)", "POMDP planners always run in real time on embedded hardware."),
]
MODELS = ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "cross-encoder/nli-deberta-v3-small", "cross-encoder/nli-deberta-v3-xsmall"]

for name in MODELS:
    print("=" * 72 + f"\n{name}")
    t = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    except Exception as e:
        print(f"  could not load: {e}")
        continue
    print(f"  load: {time.perf_counter()-t:.1f}s | labels: {model.config.id2label}")
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}

    with torch.inference_mode():
        for desc, hyp in CASES:
            enc = tok(PREMISE, hyp, return_tensors="pt", truncation="only_first", max_length=512)
            probs = model(**enc).logits.softmax(-1)[0].tolist()
            best = id2label[max(range(len(probs)), key=probs.__getitem__)]
            print(f"  {desc:40s} -> {best:13s} " + " ".join(f"{id2label[i][:7]}={p:.2f}" for i, p in enumerate(probs)))

        pairs = 64
        enc = tok([PREMISE] * pairs, [CASES[0][1]] * pairs, return_tensors="pt", padding=True,
                  truncation="only_first", max_length=512)
        t = time.perf_counter()
        for i in range(0, pairs, 16):
            model(**{k: v[i:i+16] for k, v in enc.items()})
        dt = time.perf_counter() - t
        print(f"  speed: {pairs} pairs in {dt:.2f}s  ->  {dt/pairs*1000:.0f} ms/pair (batched x16)")

print("""
HOW TO READ THIS
- 'verbatim copy' not ENTAILMENT for a model -> the app must use that model's id2label exactly as printed above.
- ms/pair here is small (<200) but the app takes ~4000 ms/pair -> the app code is the problem (reloading/no batching).
- ms/pair here is also huge -> your CPU is the bottleneck; use cross-encoder/nli-deberta-v3-xsmall.""")