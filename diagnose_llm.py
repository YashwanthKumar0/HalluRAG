import json, os, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
 
load_dotenv()
BASE, KEY, MODEL = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
print(f"Endpoint: {BASE}\nModel:    {MODEL}\n")
 
gold_path = next((p for p in [Path("data/gold/gold_qa.jsonl"), Path("data/gold/gold_qa_draft.jsonl")] if p.exists()), None)
if not gold_path:
    sys.exit("No gold file found in data/gold/")
items = [json.loads(l) for l in gold_path.open(encoding="utf-8") if l.strip()]
item = next((i for i in items if i.get("supporting_text") and i.get("answer")), None)
if not item:
    sys.exit("No gold item with supporting_text and answer found.")
 
question, passage = item["question"], item["supporting_text"]
print(f"Question: {question}\nGold answer: {item['answer']}\nPassage: {passage[:300]}\n")
 
SYSTEM = ("You are a careful research assistant. Answer the question using ONLY the numbered context passages. "
          "Cite passages like [C1]. If the context answers the question fully or partly, answer with what it supports. "
          "Only if NO passage is relevant, reply exactly: 'The provided documents do not contain enough information to answer this question.'")
USER = f"Context:\n[C1]\n{passage}\n\nQuestion: {question}\nAnswer:"
 
client = OpenAI(base_url=BASE, api_key=KEY, timeout=180)
 
def run(label, max_tokens, extra=None):
    print("=" * 70 + f"\nTEST: {label} (max_tokens={max_tokens})")
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0.1, max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
            extra_body=extra or None)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return
    msg = r.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    think = re.findall(r"<think>(.*?)(</think>|$)", content, flags=re.S)
    visible = re.sub(r"<think>.*?(</think>|$)", "", content, flags=re.S).strip()
    print(f"time: {time.perf_counter()-t:.1f}s | finish_reason: {r.choices[0].finish_reason} | usage: {r.usage}")
    print(f"has <think> block: {bool(think)} | separate reasoning field: {bool(reasoning)}")
    print(f"RAW content (first 600 chars):\n{content[:600]!r}")
    print(f"VISIBLE ANSWER:\n{visible or '(EMPTY)'}\n")
 
run("as the app runs it", 400)
run("bigger output budget", 2000)
if BASE and "11434" in BASE:
    run("Ollama with num_ctx=8192", 2000, {"options": {"num_ctx": 8192}})
USER_NO_THINK = USER + " /no_think"
print("=" * 70 + "\nTEST: thinking disabled with /no_think (works for Qwen3)")
USER, _orig = USER_NO_THINK, USER
run("no_think", 400)
 
print("=" * 70)
print("""HOW TO READ THIS
- Test 1 EMPTY + finish_reason 'length' + <think> present  -> a reasoning model is using up max_tokens. Fix: strip <think>, raise max_tokens, or use a non-thinking model.
- Test 1 answers correctly here, but the app abstains        -> the problem is in the app (retrieval/prompt building/abstain detection), not the LLM.
- Only the num_ctx test answers                            -> Ollama context truncation. Set OLLAMA_CONTEXT_LENGTH=8192.
- Every test abstains even with the exact passage          -> the model is too weak/over-cautious; switch to a stronger model.""")