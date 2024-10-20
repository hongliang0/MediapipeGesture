########################################################################################
# This script reads the landmarks from a text file and plots them on a graph.
# The text file should contain the landmarks in the following format:
# x1, y1, z1
# x2, y2, z2
# ...
# xn, yn, zn
# where x, y, and z are the coordinates of the landmarks.
# any line that cannot be converted to float will be skipped, i.e. Right Left Body.
########################################################################################


import matplotlib.pyplot as plt
import os


def read_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            try:
                x, y, z = map(float, line.split(","))
                points.append((x, -y))  # Invert the y-axis
            except ValueError:
                # Skip lines that cannot be converted to float
                continue
    return points


def plot_points(points):
    fig, ax = plt.subplots()

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    ax.scatter(xs, ys, c="b", marker="o")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")

    plt.show()


if __name__ == "__main__":
    # Change the path to the file you want to plot manually
    file_name = "../dataset/1_waving/1/1_0_landmarks.txt"
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, file_name)
    points = read_points(file_path)
    plot_points(points)
