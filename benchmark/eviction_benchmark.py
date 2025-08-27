import argparse, random, time, os, sys, shutil, glob
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import psutil
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
    """Remove all persistent cache files to ensure clean state between policy runs."""
    cleanup_paths = [
        os.path.join(ROOT, "sqlite.db"),
        os.path.join(ROOT, "faiss.index"), 
        os.path.join(ROOT, "cache_data")
    ]
    
    for path in cleanup_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

def build_data_manager(backends: str, dim: int, eviction: str, max_size: int):
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=dim)
    
    # Handle CostAware eviction policy
    if eviction == "CostAware":
        return get_data_manager(cache_base, vector_base, max_size=max_size, eviction_base="CostAware")
    else:
        return get_data_manager(cache_base, vector_base, max_size=max_size, eviction=eviction)


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

def run_workload(name, prompts, llm, policy, provider, model):
    """Run a workload and return hit rate, latency, throughput, and resource utilization."""
    hits = 0
    total_times = []
    
    # Initialize resource monitoring
    process = psutil.Process()
    memory_samples, cpu_samples = [], []
    
    for i, q in enumerate(tqdm(prompts, desc=name)):
        t0 = time.time()
        a = cache_get_compat(q)
        if a is None:
            a = llm.generate(q)  # Always use the provider's generate method
            cache_put_compat(q, a)
        else:
            hits += 1
            
        total_times.append((time.time() - t0) * 1000.0)
        
        # Sample resources every 5 requests
        if (i + 1) % 5 == 0:
            try:
                memory_samples.append(process.memory_info().rss / (1024 * 1024))  # MB
                cpu_samples.append(process.cpu_percent())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # Calculate metrics
    hit_rate = hits / len(prompts)
    latency_ms = sum(total_times) / len(total_times) if total_times else 0.0
    throughput = len(total_times) / (sum(total_times) / 1000.0) if sum(total_times) > 0 else 0.0
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

    print(f"res: hit={hit_rate:.3f} latency={latency_ms:.1f}ms throughput={throughput:.2f} mem={avg_memory_mb:.0f}MB cpu={avg_cpu_percent:.1f}% gpu={gpu_util}")
    return hit_rate, latency_ms, throughput, avg_memory_mb, avg_cpu_percent, gpu_util

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["LRU","LFU","FIFO","RR","CostAware"], 
                       choices=["LRU","LFU","FIFO","RR","CostAware"])
    parser.add_argument("--workloads", nargs="+", default=["repetitive","novel","repetitive-long","novel-long"], 
                       choices=["repetitive","novel","repetitive-long","novel-long"])
    parser.add_argument("--provider", default="dummy", choices=["dummy","ollama"])
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_faiss", action="store_true")
    parser.add_argument("--max_size", type=int, default=15)
    # Simulation is enabled by default. Use --no-simulate-latency to disable it.
    parser.add_argument("--no-simulate-latency", dest="simulate_latency", action="store_false", default=True,
                        help="Disable latency simulation for the dummy provider (useful for deterministic timing)")
    args = parser.parse_args()

    # Initialize
    random.seed(args.seed)
    # Pass simulate_latency flag through to provider (ignored by real providers)
    llm = get_provider(args.provider, model=args.model, simulate_latency=args.simulate_latency)
    onnx = Onnx()
    
    # Prepare workloads
    half = max(2, args.n // 2)
    workload_configs = {
        "repetitive": (half, 2, False), "novel": (args.n - half, 0, False),
        "repetitive-long": (half, 2, True), "novel-long": (args.n - half, 0, True)
    }
    workload_data = {name: load_mock_data(size=size, repeated=rep, isLong=long, isShuffled=True) 
                     for name, (size, rep, long) in workload_configs.items() if name in args.workloads}
    # Optional ONNX warm-up to avoid first-call initialization overhead affecting timings
    if args.warmup_iters and args.warmup_iters > 0:
        print(f"Warming up ONNX embeddings for {args.warmup_iters} iterations...")
        # Use a few short sample prompts drawn from the workloads (if available)
        warm_samples = []
        for k, v in workload_data.items():
            for i in range(min(len(v), 2)):
                warm_samples.append(v[i])
        if not warm_samples:
            warm_samples = ["warm up prompt 1", "warm up prompt 2"]
        for _ in range(args.warmup_iters):
            for sample in warm_samples:
                onnx.to_embeddings(sample)
        print("ONNX warm-up complete.")


    # Run benchmarks
    results = {}
    for policy in args.policies:
        print(f"\n=== Running policy: {policy} ===")
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
        
        # Run workloads for this policy
        results[policy] = {}
        for workload_name, data in workload_data.items():
            metrics = run_workload(f"{policy}_{workload_name}", data, llm, policy, args.provider, args.model)
            results[policy][workload_name] = dict(zip(['hit_rate', 'latency_ms', 'qps', 'avg_memory_mb', 'avg_cpu_percent', 'gpu_util'], metrics))
        
        # Clean up cache
        try:
            if hasattr(cache, 'data_manager') and cache.data_manager:
                cache.data_manager.close()
        except Exception:
            pass
    
    print_and_save_results(args, results)
    
def print_and_save_results(args, results):
    """Print summary table and append to results file."""
    # Print table
    
    header = "{:<10} {:<16} {:<8} {:<12} {:<10} {:<13} {:<8} {:<8}".format(
        'Policy', 'Workload', 'Hit Rate', 'Latency(ms)', 'Throughput', 'MemAvg(MB)', 'CPU(%)', 'GPU'
    )
    separator = "-" * len(header)
    eq_separator = "\n" + "=" * len(header)

    print(eq_separator + "\nBENCHMARK SUMMARY\n" + eq_separator)
    print(header + "\n" + separator)
    
    # Print results grouped by workload, adding a separator after each workload block
    rows = []
    for workload in args.workloads:
        printed_any = False
        for policy in args.policies:
            if workload in results.get(policy, {}):
                r = results[policy][workload]
                row = f"{policy:<10} {workload:<16} {r['hit_rate']:<8.3f} {r['latency_ms']:<12.1f} {r['qps']:<10.2f} {r['avg_memory_mb']:<13.0f} {r['avg_cpu_percent']:<8.1f} {r['gpu_util']:<8}"
                print(row)
                rows.append(row)
                printed_any = True
        # After printing all policies for this workload, add a visual separator if anything was printed
        if printed_any:
            print(separator)
            rows.append(separator)
    
    # Append to results file
    results_path = os.path.join(ROOT, "results")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = args.model if args.provider != "dummy" else "N/A"
    flags = f"policies={args.policies} workloads={args.workloads} provider={args.provider} model={model_name} n={args.n} max_size={args.max_size} warmup_iters={args.warmup_iters} seed={args.seed} use_faiss={args.use_faiss}"
    
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*90}\nTimestamp: {timestamp}\nFlags: {flags}\n\n{header}\n{separator}\n")
        f.write("\n".join(rows) + f"\n{eq_separator}\n")

    print(f"\nResults appended to: {results_path}\n{eq_separator}")

if __name__ == "__main__":
    main()


