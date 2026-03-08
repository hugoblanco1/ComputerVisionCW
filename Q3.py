from PIL import Image
import numpy as np
import timeit


def compute_colour_histogram_vectorised(img, bins=8):
    bin_map = (img.astype(np.int32) * bins) // 256
    bin_map = np.clip(bin_map, 0, bins - 1)
    bin_ids = np.arange(bins)

    hits = (bin_map[:, :, :, None] == bin_ids).astype(float)
    hist = hits.sum(axis=(0, 1))
    totals = hist.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    hist = hist / totals

    return hist


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

    hist = np.array(hist, dtype=float)

    for c in range(3):
        total = hist[c].sum()
        if total > 0:
            hist[c] = hist[c] / total

    return hist


def main():
    img = np.array(Image.open("flower.jpg").convert("RGB"))
    print(img.shape, img.dtype)

    bins_list = [8, 32, 128]
    runs = 20

    for b in bins_list:
        loop_t = timeit.timeit(lambda: compute_colour_histogram_loop(img, b), number=runs) / runs
        vec_t = timeit.timeit(lambda: compute_colour_histogram_vectorised(img, b), number=runs) / runs
        print(b, loop_t, vec_t, loop_t / vec_t)


if __name__ == "__main__":
    main()
