import json
import random
import os
import time

def load_mock_data(size=100, isLong=False, isSimilar=False, isShuffled=False, repeated=0, random_extra=-1, seed: int = None):
    """
    Load mock prompts for benchmarks.

    Returns a list of prompt strings. Each entry is a single question string.
    Adds variation so that LRU, LFU, and cost-aware policies each have strengths.

    Parameters:
    - size (int): Number of prompts to return (default 100)
    - isLong (bool): If True, use mock_data_long.json else mock_data_short.json
    - isSimilar (bool): If True, include both 'o' and 's' variants from the dataset
    - isShuffled (bool): If True, shuffle the results
    - repeated (int): Number of repetitions for each source entry
    """
    file_name = 'mock_data_long.json' if isLong else 'mock_data_short.json'
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if random_extra == -1:
        random_extra = int(0.2 * size)

    # Use a local RNG when a seed is provided to avoid mutating global random state
    rng = random.Random(seed) if seed is not None else random

    with open(file_path, 'r') as f:
        data = json.load(f)

    size = int(size - random_extra)

    result = []

    for item in data:
        # "o" is base, "s" is similar variant
        base = item['o']
        sim = item.get('s', base + "_sim")

        # Case 1: High-frequency items (good for LFU)
        if rng.random() < 0.3:
            for _ in range(repeated + rng.randint(2, 6)):
                result.append(base)

        # Case 2: Recency bursts (good for LRU)
        elif rng.random() < 0.5:
            burst_len = rng.randint(2, 4)
            burst_block = [base] * burst_len
            result.extend(burst_block)

        # Case 3: High-cost but rare items (tests cost-aware eviction)
        elif isLong and rng.random() < 0.3:
            result.append(base + "_expensive")

        # Case 4: Similar variants to confuse LFU
        if isSimilar and rng.random() < 0.4:
            result.append(sim)

        if len(result) >= size:
            break

    # Add random noise items
    while len(result) < size + random_extra:
        noise_item = rng.choice(data)['o']
        if rng.random() < 0.2 and isLong:
            noise_item += "_expensive"
        result.append(noise_item)

    if isShuffled:
        if seed is None:
            random.shuffle(result)
        else:
            rng.shuffle(result)

    return result
