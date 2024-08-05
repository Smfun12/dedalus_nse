import cv2
import os


def create_video_from_images(image_folder, output_video_path, fps=1):
    # Get list of image files in the specified directory
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure the images are sorted by name

    # Check if there are any images in the directory
    if not images:
        raise ValueError("No images found in the specified directory.")

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # Check if frame is read correctly
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping...")
            continue

        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video_path}")


# Parameters
image_folder = 'frames_nudging/'
output_video_path = 'output_video.mp4'
fps = 1  # Frames per second

# Create video
create_video_from_images(image_folder, output_video_path, fps)
