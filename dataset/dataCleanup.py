import os
import re
import numpy as np


# Function to read the file content
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Function to extract section data from the file content
def extract_section(content, section_name):
    section = re.findall(
        f"{section_name}(.*?)(Hand|Body|$)", content, re.DOTALL)
    return section[0][0].strip() if section else None


# Function to calculate the average of the first and last lines of data
def calculate_average(data_lines):
    data = [list(map(float, line.split(","))) for line in data_lines]
    avg = np.mean([data[0], data[-1]], axis=0)
    return avg


# Function to process the files
def process_file(file_path):
    content = read_file(file_path)

    # Extract Hand sections
    right_hand = extract_section(content, "Hand: Right")
    left_hand = extract_section(content, "Hand: Left")

    # If Right Hand is missing and Left Hand exists, calculate and replace it
    if not right_hand and left_hand:
        left_lines = left_hand.split("\n")
        if left_lines:  # Ensure there's actual data in Left Hand
            avg = calculate_average(left_lines)
            avg_str = ", ".join(map(str, avg))
            right_hand = f"Hand: Right\n{avg_str}"

    # If Left Hand is missing and Right Hand exists, calculate and replace it
    if not left_hand and right_hand:
        right_lines = right_hand.split("\n")
        if right_lines:  # Ensure there's actual data in Right Hand
            avg = calculate_average(right_lines)
            avg_str = ", ".join(map(str, avg))
            left_hand = f"Hand: Left\n{avg_str}"

    # If both hands are missing, you might decide to skip the file or handle it differently
    if not right_hand and not left_hand:
        print(f"Both hands are missing in file: {file_path}")
        return  # Skip further processing for this file

    # Write the cleaned content back to the file or a new file
    body = extract_section(content, "Body")
    new_content = f"{right_hand if right_hand else ''}\n\n{left_hand if left_hand else ''}\n\n{body if body else ''}"

    with open(file_path, "w") as file:
        file.write(new_content)


# Loop through all the text files in the folder
def process_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            process_file(file_path)


# Example usage:
folder_path = "./dataset/WAV3"
process_folder(folder_path)
