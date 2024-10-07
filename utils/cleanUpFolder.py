# This file deletes any files that are not .txt from a root folder and its subfolders.

import os


def delete_non_txt_files(parent_folder):
    # Hardcoded to range of folders, 1 to 50. Change the range if needed or make it dynamic.
    for folder_num in range(1, 51):
        subfolder_path = os.path.join(parent_folder, str(folder_num))
        # print(f"Checking subfolder: {subfolder_path}")
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path) and not file_name.endswith(".txt"):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
        else:
            print(f"Subfolder {folder_num} does not exist or is not a directory.")


# Replace with actual path of your parent folder
current_folder = os.path.dirname(os.path.realpath(__file__))
parent_folder_path = os.path.join(current_folder, "../dataset/5_comeHere/")
delete_non_txt_files(parent_folder_path)
