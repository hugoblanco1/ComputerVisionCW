import numpy as np
from typing import Tuple

def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            if i <= j:
                image[i][j] = 255

    print(image)
    return image


def compute_sobel_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float) / 8

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=float) / 8 

    patch = img[x-1:x+2, y-1:y+2].astype(float)

    gx = np.sum(patch * sobel_x)
    gy = np.sum(patch * sobel_y)

    mag = float(np.sqrt(gx * gx + gy * gy))
    direction = float(np.degrees(np.arctan2(gy, gx)))

    return mag, direction


def compute_diagonal_corrected_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    dx = np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ], dtype=float) / 32

    dy = np.array([
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3]
    ], dtype=float) / 32

    patch = img[x-1:x+2, y-1:y+2].astype(float)

    gx = np.sum(patch * dx)
    gy = np.sum(patch * dy)

    mag = float(np.sqrt(gx * gx + gy * gy))
    direction = float(np.degrees(np.arctan2(gy, gx)))

    return mag, direction


def main():
    image = create_diagonal_edge_image()

    sobel_mag, sobel_theta = compute_sobel_gradient(image, 4, 4)
    scharr_mag, scharr_theta = compute_diagonal_corrected_gradient(image, 4, 4)

    print("Sobel gradient:")
    print(sobel_mag, sobel_theta)

    print("Diagonal corrected gradient:")
    print(scharr_mag, scharr_theta)

    print("Improvement:")
    print(scharr_mag / sobel_mag)


if __name__ == "__main__":
    main()
