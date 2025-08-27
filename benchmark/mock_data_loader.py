import json
import random
import os

def load_mock_data(size=100, isLong=False, isSimilar=False, isMock=True, isShuffled=False, repeated=0):
    """
    Loads mock data for benchmarks based on the given parameters.

    Parameters:
    - size (int): Number of entries to return (default 100)
    - isLong (bool): If True, use mock_data_long.json, else mock_data_short.json (default False)
    - isSimilar (bool): If True, insert both "o" and "s" as separate entries, else only "o" (default False)
    - isMock (bool): If True, include "s" as the answer for each entry (default True)
    - isShuffled (bool): If True, shuffle the entries, else keep ordered (default False)
    - repeated (int): Number of repetitions for each entry (default 0)

    Returns:
    - list: List of dictionaries with "question" and "answer" keys
    """
    file_name = 'mock_data_long.json' if isLong else 'mock_data_short.json'
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    N = len(data)
    
    result = []
    for item in data:
        for _ in range(repeated + 1):
            result.append({"q": item["o"], "a": item["s"] if isMock else None})
            if len(result) >= size:
                break
        if isSimilar and len(result) < size:
            for _ in range(repeated + 1):
                result.append({"q": item["s"], "a": item["s"] if isMock else None})
                if len(result) >= size:
                    break
        if len(result) >= size:
            break
    
    if len(result) < size:
        diff = size - len(result)
        max_without_extra = N * (repeated + 1)
        if size > max_without_extra:
            print(f"Size {size} is larger than {max_without_extra} (N={N} * (repeated+1)={repeated+1}), adding {diff} extra repetitions.")
        while len(result) < size:
            entry = random.choice(result)
            result.append(entry)
    
    if isShuffled:
        random.shuffle(result)
    
    return result
