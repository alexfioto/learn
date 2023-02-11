def merge_sort(items):
    if len(items) <= 1:
        return items
    
    mid_i = len(items) // 2
    left_split = items[:mid_i]
    right_split = items[mid_i:]
    left_sorted = merge_sort(left_split)
    right_sorted = merge_sort(right_split)
    return merge(left_sorted, right_sorted)

def merge(left, right):
    result = []
    while (left and right):
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    if left:
        result += left
    if right:
        result += right
    return result

import numpy as np
unordered = np.random.randint(1, 100, 50).tolist()
print(merge_sort(unordered))