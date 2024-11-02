import shutil
import os
import time


def force_delete_folder(folder_relative_path, retries=5, delay=1):
    """Delete a folder and all of its contents with retries if files are in use."""
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.join(script_dir, folder_relative_path)

    for attempt in range(retries):
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(
                    f"The folder '{folder_path}' and all its contents have been deleted."
                )
            else:
                print(f"The folder '{folder_path}' does not exist.")
            break  # Exit loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} - Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Failed to delete the folder after multiple attempts.")


# Example usage
folder_to_delete = "../3d_dataset"
force_delete_folder(folder_to_delete)
