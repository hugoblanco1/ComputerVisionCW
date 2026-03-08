import numpy as np


def compute_inverse_transform(points, theta, t):
    pts = np.array(points, dtype=float)
    shift = np.array(t, dtype=float)

    angle = np.radians(theta)

    undo_rotation = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ])

    moved_centre = np.mean(pts, axis=0)
    original_centre = moved_centre - shift

    recovered = []

    for point in pts:
        point_back = undo_rotation @ (point - original_centre - shift) + original_centre
        recovered.append(point_back)

    return np.array(recovered)


def main():
    transformed_points = np.array([
        [3.0, 2.0],
        [2.0, 3.0],
        [4.0, 3.0]
    ])

    theta = 45
    t = [1.0, 2.0]

    original_points = compute_inverse_transform(transformed_points, theta, t)

    print("Recovered points:")
    print(original_points)


if __name__ == "__main__":
    main()
