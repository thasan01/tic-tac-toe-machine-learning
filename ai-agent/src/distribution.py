import re
import os
import json
import math
import random
from typing import Any, List, Tuple

def scan_dir(root_dir: str, file_pattern: re.Pattern) -> List[str]:
    files_to_scan = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if file_pattern.match(filename):
                files_to_scan.append(os.path.join(root, filename))
    files_to_scan.sort()
    return files_to_scan

def calc_stats(file_names: List[str], player_id: int):
    stats = {"num_sessions": len(file_names), "wins": 0, "draws": 0, "losses": 0, "files": [[], [], []]}
    for fn in file_names:
        with open(fn) as fp:
            parsed_json = json.load(fp)
            winner = parsed_json.get("winner", None)
            status_msg = parsed_json.get("status", "").lower()

            if "draw" in status_msg:
                stats["draws"] += 1
                stats["files"][1].append(fn)
            elif winner == player_id:
                stats["wins"] += 1
                stats["files"][0].append(fn)
            else:
                stats["losses"] += 1
                stats["files"][2].append(fn)
    return stats

def sample_dist(retain_dist: List[float], retain_ratio: float, data: List[List[Any]], total: int,
                rng_seed: int | None = None) -> Tuple[List[Any], List[int]]:
    """
    Performs stratified sampling to select a subset of items to retain from a categorized dataset.

    This function determines which items to keep and which to remove to achieve a
    target sample size and distribution. It first calculates the total number of items
    to retain based on an overall `retain_ratio`. It then allocates this total
    sample size across the different categories in `data` according to the proportions
    specified in `retain_dist`, while respecting the number of available items in each
    category.

    After randomly sampling the items to keep, the function returns a list of items
    that were *not* selected (i.e., the items to be removed or discarded).

    Args:
        retain_dist: A list of floats representing the target proportion of the
            final sample to be drawn from each category. Must sum to 1.0.
        retain_ratio: The overall fraction of the `total` items to retain.
        data: A list of lists, where each inner list represents a category
            and contains the items within that category.
        total: The total size of the population from which the sample is drawn,
            used with `retain_ratio` to calculate the target sample size.
        rng_seed: An optional integer to seed the random number generator for
            reproducible results.

    Returns:
        A tuple containing:
        - A list of items to be removed (the complement of the retained sample).
        - A list of integers with the final count of items retained from each
          corresponding category in `data`.
    """
    assert len(retain_dist) == len(data), "Size of `retain_dist` must match size of `data`"
    if rng_seed is not None:
        random.seed(rng_seed)

    buffer_size = total
    retain_n = int(round(retain_ratio * buffer_size))
    avail = [len(cat) for cat in data]

    desired_raw = [retain_n * r for r in retain_dist]
    desired_floor = [int(math.floor(x)) for x in desired_raw]
    remainder = [desired_raw[i] - desired_floor[i] for i in range(len(desired_raw))]

    take = [0] * len(data)

    for i in range(len(data)):
        take[i] = min(desired_floor[i], avail[i])

    slots_left = retain_n - sum(take)

    if slots_left > 0:
        idxs_by_remainder = sorted(range(len(remainder)), key=lambda x: remainder[x], reverse=True)
        for i in idxs_by_remainder:
            can_take = avail[i] - take[i]
            add = min(can_take, slots_left)
            if add > 0:
                take[i] += add
                slots_left -= add
            if slots_left == 0:
                break

    while slots_left > 0:
        remaining_idxs = [i for i in range(len(data)) if avail[i] - take[i] > 0]
        if not remaining_idxs:
            break
        total_pref = sum(retain_dist[i] for i in remaining_idxs)
        any_added = False
        for i in remaining_idxs:
            prop = (retain_dist[i] / total_pref) if total_pref > 0 else 1.0 / len(remaining_idxs)
            add = min(avail[i] - take[i], max(1, int(round(prop * slots_left))))
            if add > 0:
                take[i] += add
                slots_left -= add
                any_added = True
            if slots_left == 0:
                break
        if not any_added:
            break

    total_taken = sum(take)
    total_available = sum(avail)
    if total_available < retain_n:
        take = avail.copy()
        total_taken = total_available

    if total_taken > retain_n:
        for i in reversed(range(len(take))):
            while total_taken > retain_n and take[i] > 0:
                take[i] -= 1
                total_taken -= 1

    # Sample retained items
    retained_items = []
    for i, cnt in enumerate(take):
        if cnt > 0:
            if cnt >= len(data[i]):
                retained_items.extend(list(data[i]))
            else:
                retained_items.extend(random.sample(list(data[i]), cnt))

    # Compute items to remove = all items minus retained_items
    retained_set = set(retained_items)
    to_remove = []
    for cat in data:
        for item in cat:
            if item not in retained_set:
                to_remove.append(item)

    return to_remove, take

if __name__ == "__main__":
    data_dir = r"C:\tmp\t3\sessions"
    re_pattern = re.compile(r"training-(.*)\.txt")
    filenames = scan_dir(data_dir, re_pattern)
    stats = calc_stats(filenames, 2)
    retained, counts = sample_dist(retain_dist=[0.8, 0.1, 0.1],
                                   retain_ratio=0.5,
                                   data=stats["files"],
                                   total=stats["num_sessions"],
                                   rng_seed=42)
    print("Totals:", stats["num_sessions"], "Avail per cat:", [len(x) for x in stats["files"]])
    print("Retain counts per cat:", counts)
    print("files to remove:", retained[:10])
