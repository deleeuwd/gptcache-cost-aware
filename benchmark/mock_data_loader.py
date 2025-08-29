import json
import random
import os

def load_mock_data(size=100, isLong=False, isSimilar=False, isShuffled=False, repeated=0):
    """
    Load mock prompts for benchmarks.

    Returns a list of prompt strings. Each entry is a single question string.

    Parameters:
    - size (int): Number of prompts to return (default 100)
    - isLong (bool): If True, use mock_data_long.json else mock_data_short.json
    - isSimilar (bool): If True, include both 'o' and 's' variants from the dataset
    - isShuffled (bool): If True, shuffle the results
    - repeated (int): Number of repetitions for each source entry
    """
    file_name = 'mock_data_long.json' if isLong else 'mock_data_short.json'
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    with open(file_path, 'r') as f:
        data = json.load(f)

    result = []
    for item in data:
        for _ in range(repeated + 1):
            result.append(item['o'])
            if len(result) >= size:
                break
        if isSimilar and len(result) < size:
            for _ in range(repeated + 1):
                result.append(item['s'])
                if len(result) >= size:
                    break
        if len(result) >= size:
            break

    # If we don't have enough prompts, duplicate randomly from existing ones
    while len(result) < size:
        result.append(random.choice(result))

    if isShuffled:
        random.shuffle(result)

    return result
