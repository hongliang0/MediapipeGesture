# Written by Hongliang Sun for COMP9991 Thesis

import cv2
import mediapipe as mp
import os
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ======================
# Capture Frames
# ======================


def extractFrames(video_name):
    """Extract frames from video

    Args:
        video_name (str): video name
    """
    curr_dir = os.getcwd()
    input_dir = os.path.join(curr_dir, "raw_data", "videos")

    video_path = os.path.join(input_dir, video_name)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 2)

    count = 0
    frame_count = 0
    video_name = video_name.split(".")[0]
    output_dir = os.path.join(curr_dir, "raw_data", "frames", video_name)
    os.makedirs(output_dir, exist_ok=True)
    while True:
        success, frame = video.read()
        if not success:
            break
        if count % frame_interval == 0:
            cv2.imwrite(
                os.path.join(output_dir, f"{video_name}_{frame_count}.jpg"), frame
            )
            frame_count += 1
        count += 1

    video.release()


def parseVideos():
    curr_dir = os.getcwd()
    input_dir = os.path.join(curr_dir, "raw_data", "videos")
    for filename in os.listdir(input_dir):
        if (
            filename.endswith(".mp4")
            or filename.endswith(".avi")
            or filename.endswith(".mov")
        ):
            extractFrames(filename)


# ======================
# Inference Frames
# ======================


def captureLandmarks(image_folder):
    base_options = python.BaseOptions(model_asset_path="./models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Load the image
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = mp.Image.create_from_file(image_path)

            # Detect hand landmarks from the input image.
            detection_result = detector.detect(image)

            # Process the classification result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(
                image.numpy_view(), detection_result
            )
            file_root, file_ext = os.path.splitext(filename)
            annotated_filename = f"{file_root}_annotated{file_ext}"
            annotated_image_path = os.path.join(image_folder, annotated_filename)
            cv2.imwrite(
                annotated_image_path,
                cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
            )

            save_landmarks_to_file(image_folder, filename, detection_result)


def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def save_landmarks_to_file(image_folder, filename, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    file_root, _ = os.path.splitext(filename)
    landmark_filename = f"{file_root}_landmarks.txt"
    landmark_filepath = os.path.join(image_folder, landmark_filename)

    with open(landmark_filepath, "w") as f:
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx][0].category_name
            f.write(f"Hand: {handedness}\n")
            for landmark in hand_landmarks:
                f.write(f"{landmark.x}, {landmark.y}, {landmark.z}\n")
        # Fill with 0 if no hand landmarks detected
        if not hand_landmarks_list:
            for _ in range(21):  # 21 landmarks per hand
                f.write("0, 0, 0\n")


# ======================
# Main
# ======================

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    # Convert all videos in /raw_data/videos to frames
    # parseVideos()

    # test
    captureLandmarks("raw_data/frames/testing")
