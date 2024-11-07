import matplotlib.pyplot as plt
import os


# Updated function to read 2D points (x, y)
def read_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            try:
                x, y = map(float, line.split(","))  # Read only x and y coordinates
                points.append((x, -y))  # Invert the y-axis for plotting
            except ValueError:
                # Skip lines that cannot be converted to float
                continue
    return points


# Plot function remains unchanged
def plot_points(points):
    fig, ax = plt.subplots()

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    ax.scatter(xs, ys, c="b", marker="o")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")

    plt.show()


# Example usage
if __name__ == "__main__":
    # Change the path to the file you want to plot manually
    file_name = "../2d_dataset/1_waving/350/350_0_landmarks.txt"
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, file_name)

    # Read and plot the points
    points = read_points(file_path)
    plot_points(points)
