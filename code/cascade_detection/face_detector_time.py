import cv2
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained cascade model
cascade_path = "cascade.xml"  # The cascade file is in the same folder as the script
face_cascade = cv2.CascadeClassifier(cascade_path)

# Folder containing test images
image_folder = "image_test"  # Ensure this folder exists and contains the images
# Folder to save processed images
processed_folder = "processed_images"
os.makedirs(processed_folder, exist_ok=True)  # Create folder if it doesn't exist

# List to store data for each image
detection_data = []

# Loop through each image in the folder
for image_index, image_name in enumerate(os.listdir(image_folder), start=1):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Start timing
    start_time = time.time()

    # Perform object detection
    detected_objects = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # End timing
    process_time = time.time() - start_time

    # Number of faces detected
    num_faces = len(detected_objects)
    print(f"{num_faces} objects detected in {image_name} in {process_time:.4f} seconds")

    # Draw rectangles around detected objects
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the processed image in the new directory
    processed_image_path = os.path.join(processed_folder, image_name)
    cv2.imwrite(processed_image_path, image)

    # Add data to the table
    detection_data.append((image_index, num_faces, process_time))

# Close open windows
cv2.destroyAllWindows()

# Create a DataFrame to display the results
df = pd.DataFrame(detection_data, columns=['Image Number', 'Faces Detected', 'Processing Time (s)'])
print()
print(df)

# Exclude the first image from the statistics calculation
df_filtered = df.iloc[1:]  # Exclude the first row

# Calculate the average processing time for each number of faces detected, excluding the first image
average_times_filtered = df_filtered.groupby('Faces Detected')['Processing Time (s)'].mean()

# Convert Series to DataFrame for cleaner display
average_times_filtered_df = average_times_filtered.reset_index()
average_times_filtered_df.columns = ['Faces Detected', 'Average Processing Time (s)']
print("\nAverage processing time per number of faces detected (excluding the first image due to model loading time):")
print(average_times_filtered_df)

# Calculate the overall average detection time per image, excluding the first image
overall_average_time = df_filtered['Processing Time (s)'].mean()
print(f"\nOverall average detection time per image (excluding the first image): {overall_average_time:.4f} seconds")

# Plot the average processing time per number of faces detected
plt.figure(figsize=(10, 6))
plt.plot(average_times_filtered.index, average_times_filtered.values, marker='o')
plt.xlabel("Number of Faces Detected")
plt.ylabel("Average Processing Time (s)")
plt.title("Average Processing Time vs. Number of Faces Detected")
plt.grid(True)
plt.show()
