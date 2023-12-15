from PIL import Image
import os
import numpy as np
from app.image_processing.edge_detection import edge_detection, apply_canny, prewitt_edge_detection, log_edge_detection, apply_hed, sobel_edges
import time
import matplotlib.pyplot as plt

def evaluate_algorithm(algorithm, dataset_path):
    # Get a list of image file names in the dataset directory
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

    # Sample one image for result visualization
    sample_image_file = image_files[0]
    sample_image_path = os.path.join(dataset_path, sample_image_file)
    sample_image = Image.open(sample_image_path)

    total_runtime = 0

    for image_file in image_files:
        # Load the image from the dataset
        image_path = os.path.join(dataset_path, image_file)
        image = Image.open(image_path)

        # Measure the runtime of the edge detection algorithm
        start_time = time.time()
        result_image = algorithm(image)
        end_time = time.time()

        total_runtime += end_time - start_time

        # Ensure result_image is a PIL Image
        if not isinstance(result_image, Image.Image):
            result_image = Image.fromarray(result_image)

        # Save the result to an output directory
        output_path = os.path.join('output', f'{algorithm.__name__}_{image_file}')
        result_image.save(output_path)

    # Return the total runtime for the algorithm
    return total_runtime

if __name__ == "__main__":
    # Specify the path to the BDSD 500 dataset
    dataset_path = 'app/image_processing/BSDS500/data/images/test/'

    # Create an output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    algorithms = [edge_detection, apply_canny, prewitt_edge_detection, log_edge_detection, sobel_edges, apply_hed]
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
    sample_image = Image.open(sample_image_path)


    for algorithm in algorithms:
        # Convert the PIL Image to a NumPy array
        sample_array = np.array(sample_image)

        # Apply the edge detection algorithm to the NumPy array
        result_array = algorithm(sample_array)

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(sample_array.squeeze(), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(result_array, cmap='gray')
        plt.title(f'{algorithm.__name__}')

        plt.show()