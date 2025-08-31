# Eviction benchmark (EVICTION_BENCH.md)

This document explains how to run the eviction benchmark and describes the mock data loader used to generate workloads.

## Quick run (local)

From the project root (local environment or inside any development container):

```bash
cd benchmark
# small smoke test (fast, offline)
python eviction_benchmark.py --n 50 --warmup-iters 0 --policies LRU --workloads repetitive --provider dummy --max_size 15
```

## Command line options (short)

- `--n` : total number of requests per workload x 2 (default: 100)
- `--warmup-iters` : ONNX embedding warm-up iterations (default: 3)
- `--policies` : list of policies to test (default: LRU LFU FIFO RR CA)
- `--workloads` : list of workloads (default: repetitive novel repetitive-long novel-long)
- `--provider` : `dummy` or `ollama` (default: dummy)
- `--model` : model name passed to provider (default: llama3)
- `--use_faiss` : enable faiss vector backend (flag)
- `--max_size` : cache max size (default: 15)
- `--seed` : random seed for reproducibility (default: 42)
- `--no-simulate-latency` : disable latency simulation for dummy provider (flag)

- `--p95` : include p95 (95th percentile) latency in the printed and saved results (flag, off by default)
- `--p99` : include p99 (99th percentile) latency in the printed and saved results (flag, off by default)

## Workloads and mock data (how workloads are constructed)

The benchmark builds workloads using `benchmark/mock_data_loader.py`. The script defines a set of default workload configurations and then calls `load_mock_data(...)` to materialize each workload.

Default workload definitions (in `eviction_benchmark.py`):

```python
# Each config is: (size, repeated, isLong, isSimilar)
half = args.n // 2
workload_configs = {
	"repetitive":       (half, 3, False, True),
	"novel":            (args.n - half, 0, False, True),
	"repetitive-long":  (half, 2, True, True),
	"novel-long":       (args.n - half, 0, True, True),
}

# Then build workload_data and pass a per-workload seed for reproducibility:
workload_data = {}
for idx, (name, (size, rep, isLong, isSimilar)) in enumerate(workload_configs.items()):
	if name in args.workloads:
		workload_seed = args.seed + idx
		workload_data[name] = load_mock_data(size=size, repeated=rep, isLong=isLong, isSimilar=isSimilar, isShuffled=True, seed=workload_seed)
```

Each tuple maps to arguments passed to `load_mock_data`: `(size, repeated, isLong, isSimilar)` where:

- `size` is how many items the workload should contain
- `repeated` is how many extra copies to make for each base prompt (repetition helps create repetitive workloads)
- `isLong` selects the `mock_data_long.json` file (True) or `mock_data_short.json` (False)
- `isSimilar` controls whether similar variants (`s`) are included along with originals (`o`)

Important: the script calls `load_mock_data(..., isShuffled=True)` so workloads are shuffled by default.

What the provided default workloads mean:

- `repetitive`: half of the total requests, with `repeated=2` so prompts repeat frequently (short prompts)
- `novel`: the other half with `repeated=0` so prompts are mostly unique (short prompts)
- `repetitive-long`: same as `repetitive` but using the long prompt file
- `novel-long`: novel workload using the long prompt file

This design gives a mix of repeated vs. novel inputs and short vs. long prompts for testing eviction behavior.

## How to edit or add workloads

1. Quick change via CLI

   You can override which workloads run using the `--workloads` flag. Example to run just the novel-long workload:

```bash
python eviction_benchmark.py --workloads novel-long --n 200 --provider dummy
```

2. Edit the built-in `workload_configs` in `eviction_benchmark.py`

   Add a new entry mapping a name to `(size, repeated, isLong)`. Example to add a mixed workload that includes similar prompts:

```python
workload_configs["mixed-similar"] = (100, 1, False)

# Later, when building workload_data, pass extra params if needed:
workload_data = {}
for name, (size, rep, long) in workload_configs.items():
	 if name == "mixed-similar":
		  workload_data[name] = load_mock_data(size=size, repeated=rep, isLong=long, isSimilar=True, isShuffled=True)
	 else:
		  workload_data[name] = load_mock_data(size=size, repeated=rep, isLong=long, isShuffled=True)
```

3. Create custom workload programmatically

   You can bypass `workload_configs` entirely and construct `workload_data` yourself. Any loader that returns a list of prompt strings works.

```python
from mock_data_loader import load_mock_data
custom = load_mock_data(size=150, isLong=False, isSimilar=True, repeated=2, isShuffled=False)
# use `custom` as a workload
```

4. Add/edit source mock files

   The loader reads `benchmark/mock_data_short.json` and `benchmark/mock_data_long.json`. To add new base prompts, edit those files or add a new JSON file and modify `load_mock_data` to accept a `file_path` parameter.

## `load_mock_data` parameters (summary)

- `size` (int): number of entries to return
- `isLong` (bool): use long dataset (`mock_data_long.json`) when True
- `isSimilar` (bool): include both original (`o`) and similar (`s`) prompts as separate entries
- `isShuffled` (bool): shuffle final workload
- `repeated` (int): how many extra copies to create per base prompt (controls repetition)
- `random_extra` (int): number of extra random items added to the workload (default: -1, auto-calculated)
- `seed` (int): optional random seed to make the loader deterministic (recommended for reproducible benchmarks)

## Where to find/edit mock data

- `benchmark/mock_data_short.json` — short prompts used by default
- `benchmark/mock_data_long.json` — longer prompts used for the `*-long` workloads

To add new prompts, append entries to either JSON file. Each entry has the shape:

```json
{
	"id": 1,
	"o": "original prompt text",
	"s": "similar prompt text"
}
```

The `id` field is present for reference but is ignored by the loader. Only the `o` and `s` fields are used to generate prompts.

If you prefer separate datasets, add a new JSON file and either extend `load_mock_data` to accept a filename or pre-load that file and pass the generated list directly to the benchmark.

## Results

- The script appends a summary to the file `results` in the `benchmark/` folder.
- The console prints per-policy/workload metrics: hit rate, latency, throughput, memory and CPU usage, and optional GPU utilization.

- When invoked with `--p95` and/or `--p99` the console and appended results will also include the corresponding percentile latency columns.

## Recommended quick flags for development

- `--n 20` or `--n 50` for fast smoke tests
- `--warmup-iters 0` to skip ONNX warm-up for quicker iteration
- `--provider dummy` to avoid external LLM calls

---

# Provider file

File: `providers.py`

The benchmark supports two types of providers for generating answers to prompts:

- **DummyLLM**: A fast, local mock provider that generates synthetic answers. It can simulate latency for more realistic timing, or run instantly for deterministic tests. Use with `--provider dummy`.
- **OllamaLLM**: Connects to an external Ollama server to generate answers using real LLMs (e.g., Llama 3). Use with `--provider ollama` and set the model name with `--model`.

The provider is selected via the `--provider` flag. The benchmark uses the provider’s `generate(prompt)` method to get answers for cache misses.

# Mock data loader

File: `mock_data_loader.py`

Purpose:

- Produce synthetic benchmark inputs from `mock_data_short.json` and `mock_data_long.json`.

Function signature:

```
load_mock_data(size=100, isLong=False, isSimilar=False, isShuffled=False, repeated=0)
```

Parameters:

- `size` (int): number of entries to return.
- `isLong` (bool): choose `mock_data_long.json` (True) or `mock_data_short.json` (False).
- `isSimilar` (bool): if True, for each original prompt `o` the loader will also add the similar prompt `s` as a separate entry.
- `isShuffled` (bool): shuffle the returned list if True.
- `repeated` (int): how many extra repetitions of each prompt to insert (0 means one occurrence).

Return shape:

- `List[str]` where each item is a prompt string.

Examples:

```python
# Small set, include similar entries and shuffle
load_mock_data(size=20, isLong=False, isSimilar=True, isShuffled=True)

# Large repeated workload
load_mock_data(size=1000, isLong=True, repeated=2)
```

Implementation notes (brief):

- The loader reads the chosen JSON file (short or long), iterates entries and adds `repeated+1` copies for each original prompt `o`.
- If `isSimilar` is True it also appends the `s` prompt for each original `o` prompt.
- If requested `size` exceeds the natural maximum from the file, the function fills the remainder by sampling existing results randomly (i.e., duplicating entries).
- When `isShuffled` is True the final list is shuffled in-place.

If you need custom workloads (different similarity distributions or deterministic sequences), either modify `mock_data_loader.py` or add a new loader that returns the same `List[dict]` shape.
