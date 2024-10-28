import cv2

# Load the trained cascade model
cascade_path = "cascade.xml"  # Ensure the cascade file is available
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize video capture from the test video
video_capture = cv2.VideoCapture('big_face_test.mp4')

# Check if the video capture opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    largest_detectable_size = None
    largest_face_frame = None
    largest_face_image = None  # To store the frame with the largest face

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break loop if video has ended
        if not ret:
            print("End of video or error in capturing frame.")
            break

        # Increment frame counter
        frame_count += 1

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        detected_faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Check if any faces are detected
        if len(detected_faces) > 0:
            # Find the largest face in terms of area (width * height)
            largest_face = max(detected_faces, key=lambda rect: rect[2] * rect[3])
            (x, y, w, h) = largest_face
            face_size = w * h

            # Update largest_detectable_size if this is the largest detected so far
            if largest_detectable_size is None or face_size > largest_detectable_size:
                largest_detectable_size = face_size
                largest_face_frame = frame_count
                largest_face_image = frame.copy()  # Copy the frame

    # Draw rectangle on the largest detected face and save the image
    if largest_face_image is not None and largest_detectable_size:
        (x, y, w, h) = largest_face
        cv2.rectangle(largest_face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('largest_face_detected.png', largest_face_image)
        print(f"\nLargest detectable face size: {largest_detectable_size} pixels, detected at frame {largest_face_frame}.")
        print("Saved largest face screenshot as 'largest_face_detected.png'")
        cv2.imshow("Largest Face Detected", largest_face_image)
        cv2.waitKey(0)  # Press any key to close the image window

# Release video capture and close display window
video_capture.release()
cv2.destroyAllWindows()
