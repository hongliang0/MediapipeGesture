import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def read_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            try:
                x, y, z = map(float, line.split(","))
                points.append((x, z, -y))
            except ValueError:
                continue
    return points


def plot_points_fixed_x_rotation(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ys = [point[0] for point in points]  # This was the original X, now swapped with Y
    xs = [point[1] for point in points]  # This was the original Y, now swapped with X
    zs = [point[2] for point in points]

    scatter = ax.scatter(xs, ys, zs, c="b", marker="o")

    ax.set_xlabel("Y Label")  # X-axis now represents what used to be Y
    ax.set_ylabel("X Label")  # Y-axis now represents what used to be X
    ax.set_zlabel("Z Label")

    # Fix azimuth to a specific angle, only allowing elevation changes (rotation along X-axis restricted)
    def on_move(event):
        if event.button == 1 and event.inaxes:
            azim = ax.azim  # Fixed azimuth
            ax.view_init(
                elev=event.ydata, azim=azim
            )  # Elevation changes, azimuth remains fixed
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()


if __name__ == "__main__":
    # Change the path to the file you want to plot manually
    # file_name = "../3d_dataset/1_waving/15/15_0_landmarks.txt"
    file_name = "../test.txt"
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, file_name)
    points = read_points(file_path)
    plot_points_fixed_x_rotation(points)
