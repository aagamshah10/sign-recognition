import os
import cv2

# Directory to store the collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (categories) for which data will be collected
number_of_classes = 8

# Number of images to collect per class
dataset_size = 100

# Index of the camera to use
camera_index = 0  # Try different indices (0, 1, 2, etc.) if needed

# Open the video capture device (camera)
cap = cv2.VideoCapture(camera_index)

# Check if the camera was opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open camera with index {camera_index}")
    exit()

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Print a message indicating the current class
    print('Collecting data for class {}'.format(j))

    # Display a message to the user to prepare for data collection
    done = False
    while True:
        ret, frame = cap.read()
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print("Error: Invalid frame received.")
            continue

        # Overlay a message on the frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Display the frame
        cv2.imshow('frame', frame)
        # Check for the 'q' key press to continue to the next step
        if cv2.waitKey(25) == ord('q'):
            break

    # Counter for the number of images collected for the current class
    counter = 0
    # Loop to collect images for the current class
    while counter < dataset_size:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print("Error: Invalid frame received.")
            continue

        # Display the frame
        cv2.imshow('frame', frame)
        # Wait for a key press (for a short duration)
        cv2.waitKey(25)
        # Save the captured frame as an image file
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        # Increment the counter
        counter += 1

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
