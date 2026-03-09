from PIL import Image
import numpy as np


def patch_rgb_hist(patch: np.ndarray, bins: int = 8) -> np.ndarray:
    hist_vec = np.zeros(3 * bins, dtype=float)

    height, width = patch.shape[:2]

    for channel in range(3):
        for row in range(height):
            for col in range(width):
                pixel_val = int(patch[row, col, channel])
                bin_index = (pixel_val * bins) // 256
                hist_vec[channel * bins + bin_index] += 1.0

        start_idx = channel * bins
        end_idx = start_idx + bins
        total_vals = sum(hist_vec[start_idx:end_idx])

        if total_vals > 0:
            hist_vec[start_idx:end_idx] = hist_vec[start_idx:end_idx] / total_vals

    return hist_vec


def spatial_hist_2x2(img: np.ndarray, bins: int = 8) -> np.ndarray:
    height, width = img.shape[:2]
    mid_y = height // 2
    mid_x = width // 2

    # split image up
    regions = [
        img[0:mid_y, 0:mid_x],
        img[0:mid_y, mid_x:width],
        img[mid_y:height, 0:mid_x],
        img[mid_y:height, mid_x:width],
    ]

    final_vec = np.zeros(4 * 3 * bins, dtype=float)

    pos = 0
    for region in regions:
        region_hist = patch_rgb_hist(region, bins)
        final_vec[pos:pos + region_hist.size] = region_hist
        pos += region_hist.size

    return final_vec


def main():
    # load rgb image
    img = np.array(Image.open("flower.jpg").convert("RGB"))
    print(img.shape, img.dtype)

    # build feature vector
    feature_vec = spatial_hist_2x2(img, bins=8)
    print("Feature length:", feature_vec.size)
    print(feature_vec)


if __name__ == "__main__":
    main()
