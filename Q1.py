import numpy as np
def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
  image = np.zeros((size, size), dtype = np.uint8)

  for i in range (size):
    for j in range (size):
      if i <= j:
        image[i][j] = 255
  print(image)
  return image


image = create_diagonal_edge_image()





from typing import Tuple
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
    direction = float(np.degrees(np.arctan2(gy, gx)))  # [-180, 180]

    return mag, direction


compute_custom_gradient(image, 4, 4)



from typing import Tuple
import numpy as np

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
    direction = float(np.degrees(np.arctan2(gy, gx)))  # [-180, 180]

    return mag, direction

compute_diagonal_corrected_gradient(image, 4, 4)




