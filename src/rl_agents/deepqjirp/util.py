import numpy as np

# see https://github.com/grockious/deepsynth/blob/main/training.py
def check(a, b, upper_left):
    ul_row = upper_left[0]
    ul_col = upper_left[1]
    b_rows, b_cols = b.shape
    a_slice = a[ul_row: ul_row + b_rows, :][:, ul_col: ul_col + b_cols]
    if a_slice.shape != b.shape:
        return False
    return (a_slice == b).all()

def subarray_detector(big_array, small_array):
    upper_left = np.argwhere(big_array == small_array[0, 0])
    for ul in upper_left:
        if check(big_array, small_array, ul):
            return True
    else:
        return False
