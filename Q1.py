import numpy as np
from typing import Tuple

def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
  image = np.zeros((size, size), dtype = np.uint8)

  for i in range (size):
    for j in range (size):
      if i <= j:
        image[i][j] = 255
  print(image)
  return image


def compute_custom_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float,float]:


    dx = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]], dtype=float) / 2.0

    dy = np.array([[0, -1, 0],
                   [0,  0, 0],
                   [0,  1, 0]], dtype=float) / 2.0

    n = img[x-1:x+2, y-1:y+2].astype(float)

    gx = np.sum(n * dx)
    gy = np.sum(n * dy)
    mag = float((gx * gx + gy * gy) ** 0.5)
    direction = float(np.degrees(np.arctan2(gy, gx)))

    return mag, direction


def compute_diagonal_corrected_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:

    dx = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]], dtype=float) / 10

    dy = np.array([[-3, -10, -3],
                   [0, 0, 0],
                   [3, 10, 3]], dtype=float) / 10

    n = img[x-1:x+2, y-1:y+2].astype(float)

    gx = np.sum(n * dx)
    gy = np.sum(n * dy)

    mag = float((gx * gx + gy * gy) ** 0.5)
    direction = float(np.degrees(np.arctan2(gy, gx)))

    return mag, direction


def main():
    image = create_diagonal_edge_image()

    sobel_mag, sobel_theta = compute_custom_gradient(image, 4, 4)
    scharr_mag, scharr_theta = compute_diagonal_corrected_gradient(image, 4, 4)

    print("Custom gradient:")
    print(sobel_mag, sobel_theta)

    print("Diagonal corrected gradient:")
    print(scharr_mag, scharr_theta)

    print("Improvement:")
    print(scharr_mag / sobel_mag)


if __name__ == "__main__":
    main()
