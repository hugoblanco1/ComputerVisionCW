import numpy as np

def compute_forward_transform(points, theta, t):
  # forward transform used to test the inverse function
  
    points = np.asarray(points, dtype=float)
    t = np.asarray(t, dtype=float)

    centroid = np.mean(points, axis=0)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    transformed_points = np.dot(points - centroid, rotation.T) + centroid + t
    return transformed_points


def compute_inverse_transform(points, theta, t):
    points = np.asarray(points, dtype=float)
    t = np.asarray(t, dtype=float)
    transformed_centroid = np.mean(points, axis=0)
    original_centroid = transformed_centroid - t

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    # undo translation and undo rotation
    recovered_points = np.dot(points - transformed_centroid, rotation) + original_centroid
    return recovered_points


def main():
    points = np.array([
        [2, 9],
        [11, 1],
        [6, 7],
        [54, 2]
    ], dtype=float)

    theta = np.pi / 4
    t = np.array([3, 2], dtype=float)

    transformed = compute_forward_transform(points, theta, t)
    recovered = compute_inverse_transform(transformed, theta, t)

    print("Original points:", points)
    print("\nTransformed points:", transformed)
    print("\nRecovered points:", recovered)


if __name__ == "__main__":
    main()
