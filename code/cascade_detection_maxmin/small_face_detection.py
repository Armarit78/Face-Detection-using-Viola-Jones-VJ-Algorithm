import cv2

# Paths to input and output files
input_video_path = 'small_face_test.mp4'
output_video_path = 'full_video_small_face_test.mp4'
cascade_path = "cascade.xml"  # Path to the trained cascade model

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize video capture
video_capture = cv2.VideoCapture(input_video_path)

# Check if the video capture opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Initialize the video writer with the original frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Variables to track the smallest detected face
    frame_count = 0
    smallest_detectable_size = None
    smallest_face_frame = None
    smallest_face_image = None  # To store the frame with the smallest face
    smallest_face_coords = (0, 0, 0, 0)  # To track the smallest face coordinates

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break loop if video has ended
        if not ret:
            print("End of video or error in capturing frame.")
            break

        # Increment frame counter
        frame_count += 1

        # Write the full frame to the new video file
        video_writer.write(frame)

        # Perform face detection on the original color frame without grayscale conversion or additional parameters
        detected_faces = face_cascade.detectMultiScale(frame)

        # If faces are detected, calculate and track the smallest face size
        if len(detected_faces) > 0:
            # Find the smallest face in terms of area (width * height)
            min_face = min(detected_faces, key=lambda rect: rect[2] * rect[3])
            (x, y, w, h) = min_face
            face_size = w * h

            # Update smallest_detectable_size if this face size is the smallest detected
            if smallest_detectable_size is None or face_size < smallest_detectable_size:
                smallest_detectable_size = face_size
                smallest_face_frame = frame_count
                smallest_face_image = frame.copy()  # Copy the frame with the smallest face
                smallest_face_coords = (x, y, w, h)  # Save the smallest face coordinates

    # After processing all frames, save and display the frame with the smallest face
    if smallest_face_image is not None and smallest_detectable_size:
        # Draw rectangle on the smallest detected face
        (x, y, w, h) = smallest_face_coords
        cv2.rectangle(smallest_face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('smallest_face_detected.png', smallest_face_image)
        print(f"\nSmallest detectable face size: {smallest_detectable_size} pixels, detected at frame {smallest_face_frame}.")
        print("Saved smallest face screenshot as 'smallest_face_detected.png'")
        cv2.imshow("Smallest Face Detected", smallest_face_image)
        cv2.waitKey(0)  # Press any key to close the image window

# Release video capture and writer, then close display window
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
