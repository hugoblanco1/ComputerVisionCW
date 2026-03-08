import numpy as np


def compute_colour_histogram_vectorised(img, bins=8):
    # 1) turn each pixel value (0..255) into a bin number (0..bins-1)
    bin_map = (img.astype(np.int32) * bins) // 256
    bin_map = np.clip(bin_map, 0, bins - 1)
    bin_ids = np.arange(bins)

    hits = (bin_map[:, :, :, None] == bin_ids).astype(float)
    hist = hits.sum(axis=(0, 1))
    totals = hist.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    hist = hist / totals

    return hist


import numpy as np

def compute_colour_histogram_loop(im, num_bins=8):
    h, w, c = im.shape
    if c != 3:
        raise ValueError("Expected im to have 3 channels (H, W, 3).")

    hist = [[0] * num_bins for _ in range(3)]

    for y in range(h):
        for x in range(w):
            r = int(im[y, x, 0])
            g = int(im[y, x, 1])
            b = int(im[y, x, 2])

            hist[0][(r * num_bins) // 256] += 1
            hist[1][(g * num_bins) // 256] += 1
            hist[2][(b * num_bins) // 256] += 1

    return np.array(hist, dtype=float)

import timeit

bins_list = [8, 32, 128]
runs = 20

for b in bins_list:
    loop_t = timeit.timeit(lambda: compute_colour_histogram_loop(img, b), number=runs) / runs
    vec_t  = timeit.timeit(lambda: compute_colour_histogram_vectorised(img, b), number=runs) / runs
    print(b, loop_t, vec_t, loop_t / vec_t)
