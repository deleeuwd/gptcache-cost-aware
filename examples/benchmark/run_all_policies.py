import subprocess, sys, os
from pathlib import Path
ROOT = Path(__file__).parent
os.chdir(ROOT)                       # לרוץ מהתיקייה של הקובץ
sys.path.insert(0, str(ROOT))        # לוודא שה-import של providers עובד

CSV = "baseline.csv"
POLICIES = ["LRU","LFU","FIFO","RR"]
common = ["--n","400","--csv",CSV,"--seed","42","--provider","dummy","--use_faiss","--max_size","800"]


if os.path.exists(CSV):
    os.remove(CSV)

for p in POLICIES:
    for f in ("sqlite.db", "faiss.index", "cache.db"):
        try: os.remove(f)
        except FileNotFoundError: pass
    cmd = [sys.executable, "baseline_bench.py", "--policy", p] + common
    print(">>>"," ".join(cmd))
    subprocess.run(cmd, check=True)
