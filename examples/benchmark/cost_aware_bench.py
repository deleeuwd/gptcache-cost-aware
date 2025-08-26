import argparse, random, time
from gptcache import cache, Config
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt
from providers import get_provider
from pathlib import Path
import os, sys

# ייבוא המנגנון שלנו
from cost_aware import init_cost_aware, process_new_item, notify_hit, get_top_k_items

ROOT = Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# פונקציית עזר לאתחול נקי של ה-cache עם שם קובץ ייחודי
def init_cache(onnx_embedding, db_name: str, max_size=None, eviction_policy=None):
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    
    db_file = f"{db_name}.db"
    faiss_file = f"{db_name}.index"
    if os.path.exists(db_file): os.remove(db_file)
    if os.path.exists(faiss_file): os.remove(faiss_file)

    # שימוש בשם הקובץ הייחודי
    cache_base = CacheBase("sqlite", db_path=db_file)
    vector_base = VectorBase("faiss", dimension=onnx_embedding.dimension, index_path=faiss_file)
    
    if eviction_policy is None:
        data_manager = get_data_manager(cache_base, vector_base)
    else:
        data_manager = get_data_manager(cache_base, vector_base, max_size=max_size, eviction=eviction_policy)
    
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=onnx_embedding.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=0.95),
    )

def make_novel(n=200):
    return [f"define term_{i} in one sentence" for i in range(n)]

def run_lru_workload(name, prompts, llm):
    hits = 0
    for q in prompts:
        a = cache_get(q)
        hit = a is not None
        if not hit:
            a = llm.generate(q)
            cache_put(q, a)
        hits += int(hit)
    
    hr = hits / len(prompts)
    print(f"[{name}] hit={hr:.3f}")

def run_cost_aware_workload(name, prompts, llm, onnx_embedding, db_name):
    hits = 0
    for pid, q in enumerate(prompts):
        a = cache_get(q)
        hit = a is not None
        
        if hit:
            notify_hit(q)
        else:
            t_gen0 = time.time()
            a = llm.generate(q)
            generate_ms = (time.time() - t_gen0) * 1000.0
            
            should_rebuild = process_new_item(q, a, generate_ms)
            
            if should_rebuild:
                items_to_keep = get_top_k_items()
                # אתחול מחדש עם אותו שם קובץ ייחודי
                init_cache(onnx_embedding, db_name=db_name, max_size=None, eviction_policy=None)
                for item_q, item_a in items_to_keep:
                    cache_put(item_q, item_a)
                print(f".", end='', flush=True)
            else:
                 cache_put(q, a)

        hits += int(hit)
    
    hr = hits / len(prompts)
    print(f"\n[{name}] hit={hr:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_size", type=int, default=50)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    random.seed(42)
    onnx = Onnx()
    llm = get_provider("dummy")
    prompts = make_novel(args.n)

    # --- מבחן ראשון: LRU ---
    print("--- Running Baseline: LRU ---")
    init_cache(onnx, db_name="lru_cache", max_size=args.max_size, eviction_policy="LRU")
    run_lru_workload("LRU_novel", prompts, llm)

    # --- מבחן שני: CostAware ---
    print("\n--- Running New Policy: CostAware ---")
    db_name_cost_aware = "cost_aware_cache"
    init_cache(onnx, db_name=db_name_cost_aware, max_size=None, eviction_policy=None)
    init_cost_aware(max_size=args.max_size)
    run_cost_aware_workload("CostAware_novel", prompts, llm, onnx, db_name_cost_aware)

if __name__ == "__main__":
    main()