import os


def delete_ordered_files(base_directory):
    # Traverse each action folder in the base directory
    for action_folder in os.listdir(base_directory):
        action_folder_path = os.path.join(base_directory, action_folder)
        if os.path.isdir(action_folder_path):
            for subfolder in os.listdir(action_folder_path):
                subfolder_path = os.path.join(action_folder_path, subfolder)
                if os.path.isdir(subfolder_path) and subfolder.isdigit():
                    # Delete any file starting with "ordered"
                    for file in os.listdir(subfolder_path):
                        if file.startswith("ordered"):
                            file_path = os.path.join(subfolder_path, file)
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")


# Usage example:
# Set the base directory where all folders are located
base_directory = "./2d_dataset"
delete_ordered_files(base_directory)
