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
    """Extract exactly 10 frames from the video.

    Args:
        video_name (str): video name
    """
    curr_dir = os.getcwd()
    input_dir = os.path.join(curr_dir, "raw_data", "videos")

    video_path = os.path.join(input_dir, video_name)
    video = cv2.VideoCapture(video_path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(
        1, total_frames // 10
    )  # Calculate frame interval for 10 frames

    count = 0
    frame_count = 0
    video_name = video_name.split(".")[0]
    output_dir = os.path.join(curr_dir, "raw_data", "frames", video_name)
    os.makedirs(output_dir, exist_ok=True)

    while True:
        success, frame = video.read()
        if not success or frame_count >= 10:
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
    # Hand Landmarks
    hand_base_options = python.BaseOptions(
        model_asset_path="./models/hand_landmarker.task"
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options, num_hands=2
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # Body Landmarks
    body_base_options = python.BaseOptions(
        model_asset_path="./models/pose_landmarker_heavy.task"
    )
    body_options = vision.PoseLandmarkerOptions(
        base_options=body_base_options, output_segmentation_masks=True
    )
    body_detector = vision.PoseLandmarker.create_from_options(body_options)

    # Load the image
    for filename in os.listdir(image_folder):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".jpeg")
        ):
            image_path = os.path.join(image_folder, filename)
            image = mp.Image.create_from_file(image_path)

            # Detect hand landmarks from the input image.
            hand_detection_result = hand_detector.detect(image)
            body_detection_result = body_detector.detect(image)

            # Process the classification result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(
                image.numpy_view(), hand_detection_result, body_detection_result
            )
            file_root, file_ext = os.path.splitext(filename)
            annotated_filename = f"{file_root}_annotated{file_ext}"
            annotated_image_path = os.path.join(image_folder, annotated_filename)
            cv2.imwrite(
                annotated_image_path,
                cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
            )

            save_landmarks_to_file(
                image_folder, filename, hand_detection_result, body_detection_result
            )


# ======================
# Draw Results
# ======================


def draw_landmarks_on_image(rgb_image, hand_detection_result, body_detection_result):
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)

    hand_landmarks_list = hand_detection_result.hand_landmarks
    handedness_list = hand_detection_result.handedness
    pose_landmarks_list = body_detection_result.pose_landmarks

    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
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

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    return annotated_image


def save_landmarks_to_file(
    image_folder, filename, hand_detection_result, body_detection_result
):
    hand_landmarks_list = hand_detection_result.hand_landmarks
    handedness_list = hand_detection_result.handedness
    body_landmarks_list = body_detection_result.pose_landmarks

    file_root, _ = os.path.splitext(filename)
    landmark_filename = f"{file_root}_landmarks.txt"
    landmark_filepath = os.path.join(image_folder, landmark_filename)

    with open(landmark_filepath, "w") as f:
        if hand_landmarks_list:
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx][0].category_name
                f.write(f"Hand: {handedness}\n")
                for landmark in hand_landmarks:
                    f.write(f"{landmark.x}, {landmark.y}, {landmark.z}\n")
        else:
            for _ in range(21):  # 21 landmarks per hand
                f.write("0, 0, 0\n")

        # Write body landmarks
        if body_landmarks_list:
            for idx in range(len(body_landmarks_list)):
                f.write("Body:\n")
                for landmark in body_landmarks_list[idx]:
                    f.write(f"{landmark.x}, {landmark.y}, {landmark.z}\n")
        else:
            for _ in range(33):  # 33 landmarks for body
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
    parseVideos()

    # Capture landmarks for files in directory
    for x in range(1, 51):
        captureLandmarks(f"raw_data/frames/{x}")
