# edge_detection.py

from PIL import Image, ImageTk
from skimage import feature
import numpy as np
import cv2

def apply_canny(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=-1)

    # Apply Canny edge detection
    edge_array = feature.canny(image_array)

    # Convert the NumPy array to a PIL Image
    edge_image = Image.fromarray((edge_array * 255).astype('uint8'))

    return edge_image

def detect_edges(image_path, low_threshold=50, high_threshold=150):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Use Canny edge detector
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges

def prewitt_edge_detection(image_path, threshold=50):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Prewitt operators for horizontal and vertical gradients
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    gradient_x = cv2.filter2D(img, cv2.CV_64F, prewitt_kernel_x)
    gradient_y = cv2.filter2D(img, cv2.CV_64F, prewitt_kernel_y)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Apply thresholding to obtain binary edges
    edges = np.zeros_like(normalized_gradient)
    edges[normalized_gradient > threshold] = 255

    return edges.astype(np.uint8)

def log_edge_detection(image_path, threshold=50):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Compute gradient magnitude
    gradient_magnitude = np.abs(laplacian)

    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Apply thresholding to obtain binary edges
    edges = np.zeros_like(normalized_gradient)
    edges[normalized_gradient > threshold] = 255

    return edges.astype(np.uint8)

def sobel_edges(image_path, threshold=50):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if img is None or img.size == 0:
        print(f"Error: Unable to load image from {image_path}")
        return None

    # Apply Sobel operators for horizontal and vertical gradients
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Apply thresholding to obtain binary edges
    edges = np.zeros_like(normalized_gradient)
    edges[normalized_gradient > threshold] = 255

    return edges.astype(np.uint8)
