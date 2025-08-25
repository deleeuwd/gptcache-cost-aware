import argparse, csv, random, time
from typing import List
from gptcache import cache, Config
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt
from providers import get_provider

from pathlib import Path
import os, sys
ROOT = Path(__file__).parent
os.chdir(ROOT)                       # לרוץ מהתיקייה של הקובץ
sys.path.insert(0, str(ROOT))        # לוודא שה-import של providers עובד


def build_data_manager(backends: str, dim: int, eviction: str, max_size: int):
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=dim) if "faiss" in backends else VectorBase("faiss", dimension=dim)
    return get_data_manager(cache_base, vector_base, max_size=max_size, eviction=eviction)


def make_repetitive(n=200, k=10) -> List[str]:
    base = [f"what is {i} + {i}?" for i in range(k)]
    return [random.choice(base) for _ in range(n)]

def make_novel(n=200) -> List[str]:
    return [f"define term_{i} in one sentence" for i in range(n)]

def cache_get_compat(q: str):
    try:
        return cache_get(q)
    except TypeError:
        pass
    try:
        return cache_get(prompt=q)
    except TypeError:
        pass
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": q}]
    try:
        return cache_get(messages=msgs)
    except TypeError:
        return cache_get(prompt=q, messages=msgs)

def cache_put_compat(q: str, a: str):
    try:
        return cache_put(q, a)
    except TypeError:
        pass
    try:
        return cache_put(prompt=q, data=a)
    except TypeError:
        pass
    try:
        return cache_put(prompt=q, answer=a)
    except TypeError:
        pass
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": q}]
    try:
        return cache_put(prompt=q, data=a, messages=msgs)
    except TypeError:
        return cache_put(messages=msgs, data=a)

def run_workload(name, prompts, llm, csv_path, policy, provider, model):
    rows, hits = [], 0
    for pid, q in enumerate(prompts):
        t0 = time.time()
        t_lookup0 = time.time()
        a = cache_get_compat(q)

        lookup_ms = (time.time() - t_lookup0) * 1000.0
        hit = a is not None
        generate_ms = 0.0
        if not hit:
            t_gen0 = time.time()
            a = llm.generate(q)
            generate_ms = (time.time() - t_gen0) * 1000.0
            cache_put_compat(q, a)
        total_ms = (time.time() - t0) * 1000.0
        hits += int(hit)
        rows.append([name, pid, int(hit),
                     round(lookup_ms,3), round(generate_ms,3), round(total_ms,3),
                     len(q), len(a), provider, model, policy])
    hr = hits/len(prompts)
    gm = [r[4] for r in rows if r[2] == 0]  # generate_ms for misses only
    gen_miss_avg = sum(gm)/len(gm) if gm else 0.0
    print(f"[{name}] hit={hr:.3f}  gen_ms(miss-avg)={gen_miss_avg:.1f}")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f); ts = int(time.time())
        for r in rows: w.writerow([ts]+r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="LRU", choices=["LRU","LFU","FIFO","RR"])
    ap.add_argument("--provider", default="dummy", choices=["dummy","ollama"])
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--csv", default="baseline.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_faiss", action="store_true")
    ap.add_argument("--max_size", type=int, default=800)  # small default to trigger evictions
    args = ap.parse_args()

    random.seed(args.seed)
    onnx = Onnx()
    backends = "sqlite,faiss" if args.use_faiss else "sqlite"
    dm = build_data_manager(backends, onnx.dimension, eviction=args.policy, max_size=args.max_size)

    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=onnx.to_embeddings,
        data_manager=dm,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=0.95),  # tighter to avoid false hits
    )

    try:
        with open(args.csv,"x",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["ts","workload","prompt_id","is_hit","lookup_ms","generate_ms","total_ms",
                 "input_len","output_len","provider","model","policy"])
    except FileExistsError:
        pass

    llm = get_provider(args.provider, model=args.model)
    half = max(2, args.n//2)
    run_workload(f"{args.policy}_repetitive", make_repetitive(half, max(5,args.n//40)),
                 llm, args.csv, args.policy, args.provider, args.model)
    run_workload(f"{args.policy}_novel", make_novel(args.n-half),
                 llm, args.csv, args.policy, args.provider, args.model)

if __name__ == "__main__":
    main()


