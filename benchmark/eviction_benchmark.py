import argparse
import random
import time
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import psutil
from tqdm import tqdm
from gptcache import cache, Config
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt

from providers import get_provider
from mock_data_loader import load_mock_data


ROOT = Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def clean_cache_artifacts():
    """Remove persistent cache files to ensure clean state between runs."""
    cleanup_paths = [
        ROOT / "sqlite.db",
        ROOT / "faiss.index",
        ROOT / "cache_data",
    ]
    for path in cleanup_paths:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors


def build_data_manager(backends: str, dim: int, eviction: str, max_size: int):
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=dim)
    return get_data_manager(cache_base, vector_base, max_size=max_size, eviction=eviction)



def cache_get_compat(prompt: str):
    """Compatibility wrapper for cache.get API (handles legacy signatures)."""
    try:
        return cache_get(prompt)
    except TypeError:
        pass
    try:
        return cache_get(prompt=prompt)
    except TypeError:
        pass
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return cache_get(messages=msgs)
    except TypeError:
        return cache_get(prompt=prompt, messages=msgs)


def cache_put_compat(prompt: str, answer: str):
    """Compatibility wrapper for cache.put API (handles legacy signatures)."""
    try:
        return cache_put(prompt, answer)
    except TypeError:
        pass
    try:
        return cache_put(prompt=prompt, data=answer)
    except TypeError:
        pass
    try:
        return cache_put(prompt=prompt, answer=answer)
    except TypeError:
        pass
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return cache_put(prompt=prompt, data=answer, messages=msgs)
    except TypeError:
        return cache_put(messages=msgs, data=answer)


def run_workload(name, prompts, llm):
    """Run a workload and return performance metrics."""
    hits = 0
    latencies = []

    process = psutil.Process()
    memory_samples, cpu_samples = [], []

    for i, q in enumerate(tqdm(prompts, desc=name)):
        t0 = time.time()

        a = cache_get_compat(q)
        if a is None:
            a = llm.generate(q)
            cache_put_compat(q, a)
        else:
            hits += 1

        latencies.append((time.time() - t0) * 1000.0)

        if (i + 1) % 5 == 0:  # Sample resources periodically
            try:
                memory_samples.append(process.memory_info().rss / (1024 * 1024))  # MB
                cpu_samples.append(process.cpu_percent())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    hit_rate = hits / len(prompts)
    latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / (sum(latencies) / 1000.0) if sum(latencies) > 0 else 0.0
    avg_memory_mb = sum(memory_samples) / len(memory_samples) if memory_samples else process.memory_info().rss / (1024 * 1024)
    avg_cpu_percent = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0

    # GPU utilization (optional)
    gpu_util = "N/A"
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_util = f"{gpus[0].load * 100:.1f}%" if gpus else "N/A"
    except (ImportError, Exception):
        pass

    print(
        f"res: hit={hit_rate:.3f} latency={latency_ms:.1f}ms "
        f"throughput={throughput:.2f} mem={avg_memory_mb:.0f}MB "
        f"cpu={avg_cpu_percent:.1f}% gpu={gpu_util}"
    )
    return hit_rate, latency_ms, throughput, avg_memory_mb, avg_cpu_percent, gpu_util


def main():
    parser = argparse.ArgumentParser(description="Eviction Policy Benchmark for GPTCache")
    parser.add_argument("--policies", nargs="+", default=["LRU", "LFU", "FIFO", "RR", "CA"],
                        choices=["LRU", "LFU", "FIFO", "RR", "CA"])
    parser.add_argument("--workloads", nargs="+", default=["repetitive", "novel", "repetitive-long", "novel-long"],
                        choices=["repetitive", "novel", "repetitive-long", "novel-long"])
    parser.add_argument("--provider", default="dummy", choices=["dummy", "ollama"])
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--n", type=int, default=100, help="Number of requests per workload")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations for ONNX embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_size", type=int, default=15, help="Max cache size (entries)")
    parser.add_argument("--use-faiss", action="store_true", help="Enable Faiss vector index (default: off)")
    parser.add_argument("--no-simulate-latency", dest="simulate_latency", action="store_false", default=True,
                        help="Disable latency simulation (dummy provider only)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Initialize provider and embedding model
    # Pass seed into provider for reproducible behavior
    llm = get_provider(args.provider, model=args.model, simulate_latency=args.simulate_latency, seed=args.seed)
    onnx = Onnx()

    # Workload definitions: (size, repetitions, long, similar)
    half = args.n // 2
    workload_configs = {
        "repetitive":       (half, 3, False, True),
        "novel":            (args.n - half, 0, False, True),
        "repetitive-long":  (half, 2, True, True),
        "novel-long":       (args.n - half, 0, True, True),
    }

    # Pass seed through to the mock data loader to make workloads reproducible
    workload_data = {}
    for idx, (name, (size, rep, long, similar)) in enumerate(workload_configs.items()):
        if name in args.workloads:
            # Derive a per-workload seed so different workloads are reproducible but distinct
            workload_seed = args.seed + idx
            workload_data[name] = load_mock_data(size=size, repeated=rep, isLong=long, isSimilar=similar, isShuffled=True, seed=workload_seed)

    # Warm-up embeddings (avoid first-call overhead skewing timings)
    if args.warmup_iters > 0:
        print(f"Warming up ONNX embeddings for {args.warmup_iters} iterations...")
        warm_samples = [sample for v in workload_data.values() for sample in v[:2]]
        if not warm_samples:
            warm_samples = ["warm up prompt 1", "warm up prompt 2"]
        for _ in range(args.warmup_iters):
            for sample in warm_samples:
                onnx.to_embeddings(sample)
        print("ONNX warm-up complete.")

    # Run benchmarks
    results = {policy: {} for policy in args.policies}
    for workload_name, data in workload_data.items():
        print(f"\n=== Running workload: {workload_name} ===")
        for policy in args.policies:
            clean_cache_artifacts()

            backends = "sqlite,faiss" if args.use_faiss else "sqlite"
            dm = build_data_manager(backends, onnx.dimension, eviction=policy, max_size=args.max_size)
            
            cache.init(
                pre_embedding_func=get_prompt,
                embedding_func=onnx.to_embeddings,
                data_manager=dm,
                similarity_evaluation=SearchDistanceEvaluation(),
                config=Config(similarity_threshold=0.95),
            )

            metrics = run_workload(f"{policy}_{workload_name}", data, llm)
            results[policy][workload_name] = dict(zip(
                ["hit_rate", "latency_ms", "qps", "avg_memory_mb", "avg_cpu_percent", "gpu_util"],
                metrics
            ))

            try:
                if getattr(cache, "data_manager", None):
                    cache.data_manager.close()
            except Exception:
                pass

    print_and_save_results(args, results)


def print_and_save_results(args, results):
    """Print results table and append to file."""
    header = "{:<10} {:<16} {:<8} {:<12} {:<10} {:<13} {:<8} {:<8}".format(
        "Policy", "Workload", "Hit Rate", "Latency(ms)", "Throughput", "MemAvg(MB)", "CPU(%)", "GPU"
    )
    separator = "-" * len(header)
    eq_sep = "=" * len(header)

    print(f"\n{eq_sep}\nBENCHMARK SUMMARY\n{eq_sep}")
    print(header + "\n" + separator)

    rows = []
    for workload in args.workloads:
        for policy in args.policies:
            if workload in results.get(policy, {}):
                r = results[policy][workload]
                row = f"{policy:<10} {workload:<16} {r['hit_rate']:<8.3f} {r['latency_ms']:<12.1f} {r['qps']:<10.2f} {r['avg_memory_mb']:<13.0f} {r['avg_cpu_percent']:<8.1f} {r['gpu_util']:<8}"
                print(row)
                rows.append(row)
        print(separator)
        rows.append(separator)

    results_file = ROOT / "results"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = args.model if args.provider != "dummy" else "N/A"
    flags = f"policies={args.policies} workloads={args.workloads} provider={args.provider} model={model_name} n={args.n} max_size={args.max_size} warmup_iters={args.warmup_iters} seed={args.seed} use_faiss={args.use_faiss}"

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*90}\nTimestamp: {timestamp}\nFlags: {flags}\n\n{header}\n{separator}\n")
        f.write("\n".join(rows) + f"\n{eq_sep}\n")

    print(f"\nResults appended to: {results_file}\n{eq_sep}")


if __name__ == "__main__":
    main()
