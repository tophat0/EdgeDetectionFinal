# edge_detection.py

from PIL import Image, ImageTk
from skimage import feature, filters
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_laplace


net = cv2.dnn.readNetFromCaffe('app/image_processing/deploy.prototxt', 'app/image_processing/hed_pretrained_bsds.caffemodel')

def edge_detection(image, k_size=3, threshold=75, min_area=3):
    # Step 1: Smooth the image
    blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)

    # Step 2: Compute dark channel prior
    dark_channel = np.min(blurred, axis=2)
    dark_channel_percentile = np.percentile(dark_channel, 0.1)
    transmission = 1 - dark_channel / dark_channel_percentile

    # Step 3: Convert to grayscale
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 4: Compute gradient using Sobel operator
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Step 5: Lower the threshold for edge detection
    edges = (gradient_magnitude > threshold).astype(np.uint8) * 255

    # Step 6: Reduce the minimum contour length
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty black image to draw contours
    contour_image = np.zeros_like(gray_image)

    # Draw valid contours on the black image
    cv2.drawContours(contour_image, valid_contours, -1, (255), thickness=2)

    # Overlay contours on the original image
    result = cv2.drawContours(image.copy(), valid_contours, -1, (0, 255, 0), thickness=2)



    return edges


def apply_canny(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to the range [0, 255]
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)

    # Apply Canny edge detection
    edge_array = feature.canny(image_array)

    
    return (edge_array * 255).astype('uint8')  # Return NumPy array

def prewitt_edge_detection(image):
    # Convert the PIL Image to a NumPy array
    img = np.array(image)

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
    edges[normalized_gradient > 50] = 255

    
    return edges.astype('uint8')  # Return NumPy array

def log_edge_detection(image):
    # Convert the PIL Image to a NumPy array
    img = np.array(image)

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
    edges[normalized_gradient > 50] = 255

    return edges.astype('uint8')  # Return NumPy array

def sobel_edges(image):
    # Convert the PIL Image to a NumPy array
    img = np.array(image)

    # Apply Sobel operators for horizontal and vertical gradients
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Apply thresholding to obtain binary edges
    edges = np.zeros_like(normalized_gradient)
    edges[normalized_gradient > 50] = 255

    return edges.astype(np.uint8)

def apply_hed(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    (H, W) = image_array.shape[:2]


    blob = cv2.dnn.blobFromImage(image_array, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward() 

    # Convert the edges to a uint8 image
    hed = (hed * 255).astype(np.uint8)

    hed_image = Image.fromarray(hed.squeeze(), mode='L')

    return hed_image
