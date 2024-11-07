# Written by Hongliang Sun for COMP9991/3 Thesis

import cv2
import mediapipe as mp
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


class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(ConvLSTMModel, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim, out_channels=32, kernel_size=7, padding=3
        )  # Adjusted padding
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,  # Dropout applied between LSTM layers
        )
        self.fc_dropout = nn.Dropout(
            dropout_prob
        )  # Dropout before fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, num_files, num_points, num_features = x.size()

        # Reshape for Conv1D
        x = x.view(batch_size * num_files, num_points, num_features).permute(
            0, 2, 1
        )  # (batch_size * 10, 3, 50)

        # Apply Conv1D
        x = self.conv(
            x
        )  # Now output shape is (batch_size * 10, 32, 50) with adjusted padding
        x = x.permute(0, 2, 1)  # (batch_size * 10, 50, 32)

        # Reshape and apply LSTM
        x = x.view(batch_size, num_files, num_points, -1).mean(
            1
        )  # (batch_size, 50, 32)
        lstm_out, _ = self.lstm(x)  # (batch_size, 50, hidden_dim)

        # Apply dropout before the final fully connected layer
        final_output = self.fc_dropout(lstm_out[:, -1, :])  # (batch_size, hidden_dim)
        final_output = self.fc(final_output)  # (batch_size, output_dim)

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

    # Load the model
    input_dim = 3  # x, y, and z coordinates
    hidden_dim = 256
    num_layers = 2
    output_dim = 5  # Number of gestures
    dropout = 0.3

    model = ConvLSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    model.load_state_dict(torch.load("3d_best_model.pth"))
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

            # Only take every five frames
            if frame_counter % 2 == 0:
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
                            [landmark.x, landmark.y, landmark.z]
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
                            [landmark.x, landmark.y, landmark.z]
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
                            [landmark.x, landmark.y, landmark.z]
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
                    for landmark in necessary_body_landmarks:
                        x, y, z = landmark
                        cv2.circle(
                            image, (int(x * 640), int(y * 480)), 5, (0, 255, 0), -1
                        )

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
                    landmarks_list.append(current_frame_landmarks)

            # print(f"Current frames added to buffer {len(landmarks_list)}. Inferencing...")

            if len(landmarks_list) == 10:
                landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32).to(
                    device
                )
                # print(f"Before: {landmarks_tensor.size()}")
                landmarks_tensor = landmarks_tensor.view(1, 10, 50, 3)
                # Print out the contents of the tensor
                # print(landmarks_tensor)
                # print(f"After: {landmarks_tensor.size()}")

                output = model(landmarks_tensor)

                # DEBUG: Print the probabilities of each class
                probabilities = F.softmax(output, dim=1)
                _, predicted_gesture = torch.max(output, 1)  #

                detected_gesture = gesture_labels[predicted_gesture.item()]
                gesture_display_time = time.time()
                print(f"Detected gesture: {detected_gesture}")

                # Only keep last 9 frames
                landmarks_list = landmarks_list[1:]

            if detected_gesture and (time.time() - gesture_display_time) > 5:
                detected_gesture = None

            if probabilities is not None and (time.time() - gesture_display_time) > 1:
                probabilities = None
            elif probabilities is not None:
                y_offset = 30  # Starting position for text on the frame
                for i, prob in enumerate(probabilities[0]):
                    text = f"Class {i} ({gesture_labels[i]}): {prob.item() * 100:.2f}%"
                    cv2.putText(
                        image,
                        text,
                        (10, y_offset),  # Position on the frame
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # Font scale
                        (0, 255, 0),  # Font color (green)
                        2,  # Font thickness
                        cv2.LINE_AA,
                    )
                    y_offset += 30

            # Show the video feed with landmarks drawn
            cv2.imshow("Mediapipe Feed", image)

            frame_counter += 1

            # Exit condition
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def debug_plot_landmarks(current_frame_landmarks, frame_number):
    x = current_frame_landmarks[:, 0]
    y = current_frame_landmarks[:, 1]
    z = current_frame_landmarks[:, 2]

    # Create a 3x1 grid plot to show different views (X-Y, X-Z, Y-Z)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Single Frame 3D Landmarks Visualization", fontsize=16)

    # Define pairs to plot in each subplot
    pairs = [
        (x, y, "X-Y plane"),
        (x, z, "X-Z plane"),
        (y, z, "Y-Z plane"),
    ]

    # Plot each pair in a separate subplot
    for i, (x_data, y_data, title) in enumerate(pairs):
        axes[i].scatter(x_data, y_data, c="blue", marker="o", s=50)
        axes[i].set_title(title)
        axes[i].set_xlabel("X" if "X" in title else "Y")
        axes[i].set_ylabel("Y" if "Y" in title else "Z")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"landmarks_frame_{frame_number}.png")
    plt.close(fig)
    sys.exit()


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


def main():
    # This file is used for live-inferencing for a live video stream. It uses the model
    # trained in train.ipynb to predict the hand gestures in real-time. The model is
    # loaded from the saved model in the model directory.

    # Begin live inference
    liveCapture()


if __name__ == "__main__":
    main()
