import heapq
from collections import defaultdict

TOP_K_HEAP = []
HITS_COUNTER = defaultdict(int)
MAX_SIZE = 0
ALPHA = 1.0
BETA = 1.0

# --- פרמטרים חדשים לשליטה ב-Rebuild ---
REPLACEMENTS_SINCE_REBUILD = 0
REBUILD_THRESHOLD = 25  # לבצע Rebuild רק אחרי 25 החלפות

def init_cost_aware(max_size: int, alpha: float = 1.0, beta: float = 1.0):
    global MAX_SIZE, ALPHA, BETA, TOP_K_HEAP, HITS_COUNTER, REPLACEMENTS_SINCE_REBUILD
    MAX_SIZE = max_size
    ALPHA = alpha
    BETA = beta
    TOP_K_HEAP = []
    HITS_COUNTER = defaultdict(int)
    REPLACEMENTS_SINCE_REBUILD = 0
    print(f"[CostAwareV3] Initialized with max_size={MAX_SIZE}, rebuild_threshold={REBUILD_THRESHOLD}")

def notify_hit(q: str):
    HITS_COUNTER[q] += 1

def get_top_k_items():
    return [(q, a) for _, q, a in TOP_K_HEAP]

def calculate_score(generate_ms: float, q: str, a: str) -> float:
    size_bytes = len(q) + len(a)
    if size_bytes == 0: return 0
    hits = HITS_COUNTER.get(q, 1)
    cost = generate_ms if generate_ms > 0 else 200
    return (hits * (cost * ALPHA)) / (size_bytes * BETA)

def process_new_item(q: str, a: str, generate_ms: float):
    global REPLACEMENTS_SINCE_REBUILD
    HITS_COUNTER[q] = 1
    score = calculate_score(generate_ms, q, a)
    
    if len(TOP_K_HEAP) < MAX_SIZE:
        heapq.heappush(TOP_K_HEAP, (score, q, a))
        return False
    
    worst_score, _, _ = TOP_K_HEAP[0]
    if score > worst_score:
        heapq.heapreplace(TOP_K_HEAP, (score, q, a))
        REPLACEMENTS_SINCE_REBUILD += 1
        if REPLACEMENTS_SINCE_REBUILD >= REBUILD_THRESHOLD:
            REPLACEMENTS_SINCE_REBUILD = 0 # איפוס המונה
            return True # הגיע הזמן לבצע Rebuild
    
    return False