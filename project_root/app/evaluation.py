from PIL import Image
import os
import numpy as np
import cv2
from image_processing.edge_detection import edge_detection, apply_canny, prewitt_edge_detection, log_edge_detection, apply_hed, apply_sobel, apply_kirsch
import time
import matplotlib.pyplot as plt

# This function will take an edge detection algoritm and for each image in a directory, will test that edge detection method on those images and track runtime.
def evaluate_algorithm(algorithm, dataset_path):
    # Get a list of image file names in the dataset directory
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

    # Sample one image for result visualization
    sample_image_file = image_files[0]
    sample_image_path = os.path.join(dataset_path, sample_image_file)
    sample_image = cv2.imread(sample_image_path)

    total_runtime = 0

    for image_file in image_files:
        # Load the image from the dataset
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)

        # Measure the runtime of the edge detection algorithm
        start_time = time.time()
        result_image = algorithm(image)
        end_time = time.time()

        # Add time to evaluate the image to total runtime
        total_runtime += end_time - start_time

        # Ensure result_image is a NumPy array
        if isinstance(result_image, Image.Image):
            result_image = np.array(result_image)

        # Save the result(s) to an output directory
        if isinstance(result_image, tuple): #If it is our edge detection method (2 results are returned), save to output
            edges_output_path = os.path.join('output', f'{algorithm.__name__}_edges_{image_file}')
            cv2.imwrite(edges_output_path, result_image[0])

            contour_output_path = os.path.join('output', f'{algorithm.__name__}_contour_{image_file}')
            cv2.imwrite(contour_output_path, result_image[1])       

    # Return the total runtime for the algorithm
    return total_runtime

if __name__ == "__main__":
    # Specify the path to the BDSD 500 dataset
    dataset_path = 'app/image_processing/BSDS500/data/images/test/'

    # Send images to output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Algorithms to evaluate
    algorithms = [edge_detection, apply_canny, prewitt_edge_detection, log_edge_detection, apply_sobel, apply_kirsch, apply_hed]
    runtimes = []

    # Evaluate each edge detection algorithm and collect runtimes
    for algorithm in algorithms:
        runtime = evaluate_algorithm(algorithm, dataset_path)
        runtimes.append(runtime)
        print(f'{algorithm.__name__} - Total Runtime: {runtime:.4f} seconds')


    # Plot the runtimes
    plt.figure(figsize=(10, 6))
    plt.bar([algo.__name__ for algo in algorithms], runtimes)
    plt.xlabel('Edge Detection Algorithm')
    plt.ylabel('Total Runtime (seconds)')
    plt.title('Runtimes of Edge Detection Algorithms')
    plt.show()

    
    # Show the result images for one sample image
    sample_image_file = '201080.jpg'
    sample_image_path = os.path.join(dataset_path, sample_image_file)
    sample_image = cv2.imread(sample_image_path)


    for algorithm in algorithms:
        # Convert the PIL Image to a NumPy array
        sample_array = np.array(sample_image)

        # Apply the edge detection algorithm to the NumPy array
        result_array = algorithm(sample_array)

        plt.figure(figsize=(8, 4))

        if isinstance(result_array, tuple):
            # Display the edges image
            plt.subplot(1, 2, 1)
            plt.imshow(result_array[0], cmap='gray')
            plt.title(f'{algorithm.__name__} - Edges')

            # Display the contour image
            plt.subplot(1, 2, 2)
            plt.imshow(result_array[1], cmap='gray')
            plt.title(f'{algorithm.__name__} - Contour')
        else:
            # Display the single result image
            plt.imshow(result_array, cmap='gray')
            plt.title(f'{algorithm.__name__}')

        plt.show()