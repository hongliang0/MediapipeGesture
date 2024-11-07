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


class EnhancedConvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(EnhancedConvLSTMModel, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # Second Conv layer
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )

        # Dropout and fully connected layer
        self.fc_dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, num_files, num_points, num_features = x.size()

        # Reshape for Conv1D
        x = x.view(batch_size * num_files, num_points, num_features).permute(0, 2, 1)

        # Apply Conv1D layers with BatchNorm and ReLU
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        # Reshape for LSTM input
        x = x.permute(0, 2, 1).view(batch_size, num_files, num_points, -1).mean(1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Final output with dropout and fully connected layer
        final_output = self.fc_dropout(lstm_out[:, -1, :])
        final_output = self.fc(final_output)

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

    # Define model with updated parameters
    input_dim = 2  # x and y coordinates
    hidden_dim = 256
    num_layers = 2
    output_dim = 5  # Number of gestures
    dropout = 0.3

    model = EnhancedConvLSTMModel(
        input_dim, hidden_dim, num_layers, output_dim, dropout
    )
    model.load_state_dict(torch.load("2d_best_model.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Mediapipe drawing and pose utilities
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
    smooth_frames = 5

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
                    # print("Right hand:\n", right_hand_landmarks)
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
                    # print("Body:\n", body_landmarks)

                    # Process body landmarks as we dont keep all
                    # Step 2: Split the remaining lines into hand data and body data
                    necessary_body_landmarks = np.concatenate(
                        (body_landmarks[11:17], body_landmarks[23:25]), axis=0
                    )
                    current_frame_landmarks.extend(necessary_body_landmarks)

                    # Plot the necessary body landmarks as test
                    # for landmark in necessary_body_landmarks:
                    #     x, y, z = landmark
                    #     cv2.circle(
                    #         image, (int(x * 640), int(y * 480)), 5, (0, 255, 0), -1
                    #     )

                # Check if our landmarks are ok, i.e. all detections critical found
                # print(len(current_frame_landmarks))
                if len(current_frame_landmarks) == 50:
                    # print(current_frame_landmarks)
                    current_frame_landmarks = np.array(current_frame_landmarks)
                    # if current_frame_landmarks.shape == (50, 3):
                    #     print("The shape is correct:", current_frame_landmarks.shape)
                    # else:
                    #     print("Unexpected shape:", current_frame_landmarks.shape)
                    # Process model
                    # current_frame_landmarks = np.array(current_frame_landmarks).reshape(
                    #     50, 3
                    # )
                    # debug_save_landmarks_to_txt(current_frame_landmarks, frame_counter)
                    # debug_plot_landmarks(current_frame_landmarks, frame_counter)
                    landmarks_list.append(
                        reorder_landmarks(current_frame_landmarks, order="distance")
                    )

            # print(f"Current frames added to buffer {len(landmarks_list)}. Inferencing...")

            if len(landmarks_list) == 10:
                landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32).to(
                    device
                )
                # print(f"Before: {landmarks_tensor.size()}")
                landmarks_tensor = landmarks_tensor.view(1, 10, 50, 2)
                # Print out the contents of the tensor
                # print(landmarks_tensor)
                # print(f"After: {landmarks_tensor.size()}")

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
                print(f"Detected gesture: {detected_gesture}")

                # Only keep last 9 frames
                # landmarks_list = landmarks_list[1:]
                landmarks_list = []

            if detected_gesture and (time.time() - gesture_display_time) > 5:
                detected_gesture = None

            # Display gesture and probabilities if available
            if probabilities is not None and (time.time() - gesture_display_time) <= 1:
                # Display detected gesture in the top-left corner
                if detected_gesture:
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
    return landmarks


def main():
    # This file is used for live-inferencing for a live video stream. It uses the model
    # trained in train.ipynb to predict the hand gestures in real-time. The model is
    # loaded from the saved model in the model directory.

    # Begin live inference
    liveCapture()


if __name__ == "__main__":
    main()
