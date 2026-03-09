from PIL import Image
import numpy as np
import timeit


def compute_colour_histogram_vectorised(img, num_bins=8):
    bin_map = (img.astype(np.int32) * num_bins) // 256
    bin_map = np.clip(bin_map, 0, num_bins - 1)
    bin_ids = np.arange(num_bins)

    # compare against bins
    hits = (bin_map[:, :, :, None] == bin_ids).astype(float)

    hist = hits.sum(axis=(0, 1))
    totals = hist.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    hist = hist / totals

    return hist


def compute_colour_histogram_loop(img, num_bins=8):
    height, width, channels = img.shape

    if channels != 3:
        raise ValueError("Expected image shape (H, W, 3).")

    hist = [[0] * num_bins for _ in range(3)]

    for y in range(height):
        for x in range(width):
            r = int(img[y, x, 0])
            g = int(img[y, x, 1])
            b = int(img[y, x, 2])

            hist[0][(r * num_bins) // 256] += 1
            hist[1][(g * num_bins) // 256] += 1
            hist[2][(b * num_bins) // 256] += 1

    hist = np.array(hist, dtype=float)

    for channel in range(3):
        total = hist[channel].sum()
        if total > 0:
            hist[channel] = hist[channel] / total

    return hist


def main():
    img = np.array(Image.open("flower.jpg").convert("RGB"))
    print(img.shape, img.dtype)

    bins_list = [8, 32, 128]
    runs = 20

    for num_bins in bins_list:
        loop_time = timeit.timeit(
            lambda: compute_colour_histogram_loop(img, num_bins),
            number=runs
        ) / runs

        vec_time = timeit.timeit(
            lambda: compute_colour_histogram_vectorised(img, num_bins),
            number=runs
        ) / runs

        print(num_bins, loop_time, vec_time, loop_time / vec_time)


if __name__ == "__main__":
    main()
