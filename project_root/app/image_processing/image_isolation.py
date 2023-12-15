import cv2
import numpy as np

def isolate_and_count_objects(edge_image, num_objects_to_display=5, threshold_value=50):
    try:
        # Threshold the edge image to obtain a binary image
        _, binary_image = cv2.threshold(edge_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Initialize a list to store cutout images
        cutout_images = []

        # Iterate through selected contours and extract cutout images
        for i, contour in enumerate(sorted_contours[:num_objects_to_display]):
            # Create a mask for each object
            mask = np.zeros_like(edge_image)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            # Extract the cutout image using the mask
            cutout = cv2.bitwise_and(edge_image, edge_image, mask=mask)

            # Add the cutout image to the list
            cutout_images.append(cutout)

        # Print and return the number of objects and cutout images
        total_objects = len(contours)
        num_objects = len(cutout_images)
        print(f"Total number of objects: {total_objects}")
        print(f"Number of objects to display: {num_objects}")

        return total_objects, num_objects, cutout_images

    except Exception as e:
        print(f"Error in isolate_and_count_objects: {e}")
        return None, None, None
