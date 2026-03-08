from PIL import Image
import numpy as np


def patch_rgb_hist(patch: np.ndarray, bins: int = 8) -> np.ndarray:
    hist = np.zeros(3 * bins, dtype=float)

    h, w = patch.shape[:2]

    for c in range(3):
        for i in range(h):
            for j in range(w):
                v = int(patch[i, j, c])
                b = (v * bins) // 256
                hist[c * bins + b] += 1.0

        start = c * bins
        end = start + bins
        total = sum(hist[start:end])

        if total > 0:
            hist[start:end] = hist[start:end] / total

    return hist


def spatial_hist_2x2(img: np.ndarray, bins: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    yc = h // 2
    xc = w // 2

    patches = [
        img[0:yc, 0:xc],
        img[0:yc, xc:w],
        img[yc:h, 0:xc],
        img[yc:h, xc:w],
    ]

    feat = np.zeros(4 * 3 * bins, dtype=float)

    pos = 0
    for p in patches:
        hv = patch_rgb_hist(p, bins)
        feat[pos:pos + hv.size] = hv
        pos += hv.size

    return feat


def main():
    img = np.array(Image.open("flower.jpg").convert("RGB"))
    print(img.shape, img.dtype)

    feature_vec = spatial_hist_2x2(img, bins=8)
    print("Feature length:", feature_vec.size)
    print(feature_vec)


if __name__ == "__main__":
    main()
