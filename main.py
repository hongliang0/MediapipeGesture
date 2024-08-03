# Written by Hongliang Sun for COMP9991 Thesis

import cv2
import mediapipe as mp
import os
import time


# ======================
# Save Coordinates
# ======================


def update_next_file_number(cache_file_path):
    """
    Update the file number stored in a cache file.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        int: The next file number.
    """
    try:
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                current_number = int(file.read().strip())
        else:
            current_number = 0

        next_number = current_number + 1

        with open(cache_file_path, "w") as file:
            file.write(str(next_number))

        return next_number

    except Exception as e:
        print(f"Error updating file number: {e}")
        return None


def get_current_file_number(cache_file_path):
    """
    Read the current file number from the cache file.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        int: The current file number.
    """
    try:
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                return int(file.read().strip())
        return 0  # Default to 0 if no cache file exists
    except Exception as e:
        print(f"Error reading file number: {e}")
        return None


def save_coordinates_wave(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """
    Function to save coordinates to file

    Args:
        pose_landmarks (object): Data structure containing pose landmarks.
        left_hand_landmarks (object): Data structure containing left hand landmarks.
        right_hand_landmarks (object): Data structure containing right hand landmarks.
    """
    dataset_location = "dataset/1_waving/"
    cache_file_path = f"{dataset_location}cache_file.txt"
    prefix = "waving"
    extension = "csv"
    number = get_current_file_number(cache_file_path)

    for frame in range(20):
        filename = f"{dataset_location}{prefix}{number}_{frame}.{extension}"
        with open(filename, "w") as f:
            if pose_landmarks:
                f.write("Upper Body Landmarks:\n")
                for id, lm in enumerate(pose_landmarks.landmark):
                    f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
            if left_hand_landmarks:
                f.write("Left Hand Landmarks:\n")
                for id, lm in enumerate(left_hand_landmarks.landmark):
                    f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
            if right_hand_landmarks:
                f.write("Right Hand Landmarks:\n")
                for id, lm in enumerate(right_hand_landmarks.landmark):
                    f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        time.sleep(1)


def save_coordinates_thumbs_up(
    pose_landmarks, left_hand_landmarks, right_hand_landmarks
):
    """
    Function to save coordinates to file

    Args:
        pose_landmarks (object): Data structure containing pose landmarks.
        left_hand_landmarks (object): Data structure containing left hand landmarks.
        right_hand_landmarks (object): Data structure containing right hand landmarks.
    """
    with open("coordinates.txt", "w") as f:
        if pose_landmarks:
            f.write("Pose Landmarks:\n")
            for id, lm in enumerate(pose_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if left_hand_landmarks:
            f.write("Left Hand Landmarks:\n")
            for id, lm in enumerate(left_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if right_hand_landmarks:
            f.write("Right Hand Landmarks:\n")
            for id, lm in enumerate(right_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")


def save_coordinates_pointing(
    pose_landmarks, left_hand_landmarks, right_hand_landmarks
):
    """
    Function to save coordinates to file

    Args:
        pose_landmarks (object): Data structure containing pose landmarks.
        left_hand_landmarks (object): Data structure containing left hand landmarks.
        right_hand_landmarks (object): Data structure containing right hand landmarks.
    """
    with open("coordinates.txt", "w") as f:
        if pose_landmarks:
            f.write("Pose Landmarks:\n")
            for id, lm in enumerate(pose_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if left_hand_landmarks:
            f.write("Left Hand Landmarks:\n")
            for id, lm in enumerate(left_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if right_hand_landmarks:
            f.write("Right Hand Landmarks:\n")
            for id, lm in enumerate(right_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")


def save_coordinates_shrugging(
    pose_landmarks, left_hand_landmarks, right_hand_landmarks
):
    """
    Function to save coordinates to file

    Args:
        pose_landmarks (object): Data structure containing pose landmarks.
        left_hand_landmarks (object): Data structure containing left hand landmarks.
        right_hand_landmarks (object): Data structure containing right hand landmarks.
    """
    with open("coordinates.txt", "w") as f:
        if pose_landmarks:
            f.write("Pose Landmarks:\n")
            for id, lm in enumerate(pose_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if left_hand_landmarks:
            f.write("Left Hand Landmarks:\n")
            for id, lm in enumerate(left_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if right_hand_landmarks:
            f.write("Right Hand Landmarks:\n")
            for id, lm in enumerate(right_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")


def save_coordinates_come_here(
    pose_landmarks, left_hand_landmarks, right_hand_landmarks
):
    """
    Function to save coordinates to file

    Args:
        pose_landmarks (object): Data structure containing pose landmarks.
        left_hand_landmarks (object): Data structure containing left hand landmarks.
        right_hand_landmarks (object): Data structure containing right hand landmarks.
    """
    with open("coordinates.txt", "w") as f:
        if pose_landmarks:
            f.write("Pose Landmarks:\n")
            for id, lm in enumerate(pose_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if left_hand_landmarks:
            f.write("Left Hand Landmarks:\n")
            for id, lm in enumerate(left_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")
        if right_hand_landmarks:
            f.write("Right Hand Landmarks:\n")
            for id, lm in enumerate(right_hand_landmarks.landmark):
                f.write(f"{id}: ({lm.x}, {lm.y}, {lm.z})\n")


# ======================
# Capture Frames
# ======================


def liveCapture():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose, mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process image and find landmarks
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            holistic_results = holistic.process(image_rgb)

            # Draw pose and hand landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            if holistic_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    holistic_results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )
            if holistic_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    holistic_results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )

            # Display the image
            cv2.imshow("Posture and Hand Detection", image)

            # Save coordinates when keys are pressed.
            # 1 - waving
            # 2 - thumbs up
            # 3 - pointing
            # 4 - shrugging
            # 5 - come here
            if cv2.waitKey(5) & 0xFF == ord("1"):
                save_coordinates_wave(
                    pose_results.pose_landmarks,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )
            if cv2.waitKey(5) & 0xFF == ord("2"):
                save_coordinates_thumbs_up(
                    pose_results.pose_landmarks,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )
            if cv2.waitKey(5) & 0xFF == ord("3"):
                save_coordinates_pointing(
                    pose_results.pose_landmarks,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )
            if cv2.waitKey(5) & 0xFF == ord("4"):
                save_coordinates_shrugging(
                    pose_results.pose_landmarks,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )
            if cv2.waitKey(5) & 0xFF == ord("5"):
                save_coordinates_come_here(
                    pose_results.pose_landmarks,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )

            # Exit when 'ESC' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    liveCapture()
