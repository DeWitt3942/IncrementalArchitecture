def str_array(arr, mapping = None):
    if not (mapping is None):
        arr = map(mapping, arr)
    return "["+ " ".join(map(str, arr))+" ]"

def round3(number):
    return round(number=number, ndigits=3)

import numpy as np


def k_most(arr, k):
	if k==0:
		return []
	return np.argpartition(arr,  -k)[-k:]
