# Written by Hongliang Sun for COMP9991/3 Thesis

import cv2
import mediapipe as mp
import math
import numpy as np
import os
import time
import sys

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


import torch
import torch.nn as nn


class RefinedConvModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.3):
        super(RefinedConvModel, self).__init__()

        # First Conv layer - process the 2D coordinates
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # Second Conv layer
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Third Conv layer (optional, but keeps the feature extraction strong)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        # Dropout layer
        self.fc_dropout = nn.Dropout(dropout_prob)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # x shape: (batch_size, num_files, num_points, num_features)
        batch_size, num_files, num_points, num_features = x.size()

        # Reshape for Conv1D: (batch_size * num_files, num_features, num_points)
        x = x.view(batch_size * num_files, num_points, num_features).permute(0, 2, 1)

        # Apply Conv1D layers with BatchNorm and ReLU
        x = self.relu1(
            self.bn1(self.conv1(x))
        )  # (batch_size * num_files, 64, num_points)
        x = self.relu2(
            self.bn2(self.conv2(x))
        )  # (batch_size * num_files, 128, num_points)
        x = self.relu3(
            self.bn3(self.conv3(x))
        )  # (batch_size * num_files, 256, num_points)

        # Global Average Pooling over the time dimension
        x = x.mean(dim=2)  # Shape: (batch_size * num_files, 256)

        # Reshape back to (batch_size, num_files, 256) and apply dropout
        x = x.view(batch_size, num_files, -1)
        x = self.fc_dropout(x)

        # Apply the final fully connected layer to each sequence independently
        final_output = self.fc(
            x.mean(dim=1)
        )  # Global average pooling across sequences for classification

        return final_output


def liveCapture():
    # Load args

    gesture_labels = [
        "Waving",
        "Thumbs Up",
        "Pointing",
        "Shrugging",
        "Come Here",
    ]

    input_dim = 2  # x and y coordinates
    output_dim = 5  # Number of gestures
    dropout_prob = 0.3
    model = RefinedConvModel(input_dim, output_dim, dropout_prob)
    model.load_state_dict(torch.load("2d_best_model.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialise Mediapipe drawing and pose utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    frame_counter = 0  # Change this dynamically if laptop does not perform as well as expectations...
    landmarks_list = []  # Store our landmarks here which well feed into model
    detected_gesture = None
    probabilities = None
    prob_history = []
    smooth_frames = 5  # Average the probabilities over the last 5 frames
    cached_refresh = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Flip and convert the image color to RGB for Mediapipe processing
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Convert image back to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Only take every two frames (?)
            if frame_counter % 1 == 0:
                current_frame_landmarks = []
                # Draw and print left hand landmarks. Remember left hand first!
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    )
                    left_hand_landmarks = np.array(
                        [
                            [landmark.x, landmark.y]
                            for landmark in results.left_hand_landmarks.landmark
                        ]
                    )
                    current_frame_landmarks.extend(left_hand_landmarks)

                # Draw and print right hand landmarks. Remember right hand second!
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    )
                    right_hand_landmarks = np.array(
                        [
                            [landmark.x, landmark.y]
                            for landmark in results.right_hand_landmarks.landmark
                        ]
                    )
                    current_frame_landmarks.extend(right_hand_landmarks)

                # Draw and print pose landmarks. Remember body last!
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    body_landmarks = np.array(
                        [
                            [landmark.x, landmark.y]
                            for landmark in results.pose_landmarks.landmark
                        ]
                    )

                    # Step 2: Split the remaining lines into hand data and body data
                    necessary_body_landmarks = np.concatenate(
                        (body_landmarks[11:17], body_landmarks[23:25]), axis=0
                    )
                    current_frame_landmarks.extend(necessary_body_landmarks)

                # Check if our landmarks are ok, i.e. all detections critical found
                if len(current_frame_landmarks) == 50:
                    # print(current_frame_landmarks)
                    current_frame_landmarks = np.array(current_frame_landmarks)
                    normalized_landmarks = normalize_points(current_frame_landmarks)
                    landmarks_list.append(normalized_landmarks)
                    cached_refresh = 0
                else:
                    cached_refresh += 1

            # Implement cached refresh to autoclear frames from buffer
            if cached_refresh > 5:
                cached_refresh = 0
                landmarks_list = []
                landmarks_tensor = None

            if len(landmarks_list) == 10:
                landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32).to(
                    device
                )
                landmarks_tensor = landmarks_tensor.view(1, 10, 50, 2)
                output = model(landmarks_tensor)

                # DEBUG: Print the probabilities of each class
                probabilities = F.softmax(output, dim=1)
                prob_history.append(probabilities[0].cpu().detach().numpy())
                if len(prob_history) > smooth_frames:
                    prob_history.pop(0)

                avg_probabilities = np.mean(prob_history, axis=0)
                predicted_gesture = np.argmax(avg_probabilities)
                detected_gesture = gesture_labels[predicted_gesture.item()]
                gesture_display_time = time.time()

                landmarks_list = []
                landmarks_tensor = None
                prob_history = []

            if detected_gesture and (time.time() - gesture_display_time) > 5:
                detected_gesture = None

            # Display gesture and probabilities if available
            if probabilities is not None and (time.time() - gesture_display_time) <= 1:
                # Display detected gesture in the top-left corner
                if detected_gesture and avg_probabilities[predicted_gesture] > 0.5:
                    cv2.putText(
                        image,
                        f"Gesture: {detected_gesture}",
                        (10, 30),  # Position at the top-left corner
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,  # Larger font scale for emphasis
                        (0, 255, 0),  # Font color (green)
                        2,  # Font thickness
                        cv2.LINE_AA,
                    )

                # Display probabilities in the top-right corner
                y_offset = 30  # Start offset for probabilities
                x_offset = (
                    image.shape[1] - 200
                )  # Right-aligned position for probabilities

                # Use avg_probabilities for smoother display
                for i, prob in enumerate(avg_probabilities):
                    text = f"{gesture_labels[i]}: {prob * 100:.2f}%"
                    cv2.putText(
                        image,
                        text,
                        (x_offset, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # Smaller font scale
                        (0, 255, 0),  # Font color (green)
                        1,  # Font thickness
                        cv2.LINE_AA,
                    )
                    y_offset += 20  # Move down for each probability line

            # Show the video feed with landmarks drawn
            cv2.imshow("Mediapipe Feed", image)

            frame_counter += 1

            # Exit condition
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def normalize_points(points):
    # Calculate the centroids for x, y, and z
    centroid_x = sum(point[0] for point in points) / len(points)
    centroid_y = sum(point[1] for point in points) / len(points)

    # Normalize points by centering around the centroids
    normalized_points = [(x - centroid_x, y - centroid_y) for x, y in points]

    return normalized_points


def debug_save_landmarks_to_txt(current_frame_landmarks, frame_number):
    # Convert the landmarks array to ensure it's a NumPy array
    current_frame_landmarks = np.array(current_frame_landmarks)

    # Define the filename
    filename = f"landmarks_frame_{frame_number}.txt"

    # Save each (x, y, z) point on a new line
    with open(filename, "w") as file:
        for point in current_frame_landmarks:
            x, y, z = point
            file.write(f"{x}, {y}, {z}\n")
    print(f"Landmarks saved to {filename}")


def reorder_landmarks(landmarks, order="distance"):
    # Ensure landmarks is a list
    if not isinstance(landmarks, list):
        landmarks = (
            landmarks.tolist()
        )  # Convert to list if it's a NumPy array or similar

    if order == "x":
        # Sort landmarks by x coordinate
        landmarks.sort(key=lambda p: p[0])
    elif order == "y":
        # Sort landmarks by y coordinate
        landmarks.sort(key=lambda p: p[1])
    elif order == "distance":
        # Sort landmarks by distance from the origin (0, 0)
        landmarks.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5)
    elif order == "angle":
        # Sort landmarks by angle relative to the centroid
        centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
        centroid_y = sum(point[1] for point in landmarks) / len(landmarks)
        landmarks.sort(key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))
    # print(landmarks)
    return landmarks


def main():
    # This file is used for live-inferencing for a live video stream. It uses the model
    # trained in train.ipynb to predict the hand gestures in real-time. The model is
    # loaded from the saved model in the model directory.

    # Begin live inference
    liveCapture()


if __name__ == "__main__":
    main()
