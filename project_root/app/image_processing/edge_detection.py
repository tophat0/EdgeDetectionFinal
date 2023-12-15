# edge_detection.py

from PIL import Image, ImageTk
from skimage import feature, filters
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_laplace


net = cv2.dnn.readNetFromCaffe('app/image_processing/deploy.prototxt', 'app/image_processing/hed_pretrained_bsds.caffemodel')

def edge_detection(image):
    image_array = np.array(image)
    # Step 1: Smooth the image
    blurred = cv2.GaussianBlur(image_array, (5, 5), 0)

    # Step 2: Compute dark channel prior
    dark_channel = np.min(blurred, axis=2)
    dark_channel_percentile = np.percentile(dark_channel, 0.1)
    transmission = 1 - dark_channel_percentile

    # Step 3: Convert to grayscale
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 4: Compute gradient using Sobel operator
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Step 5: Lower the threshold for edge detection
    edges = (gradient_magnitude > 75).astype(np.uint8) * 255

    # Step 6: Reduce the minimum contour length
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 3]

    # Create an empty black image to draw contours
    contour_image = np.zeros_like(gray_image)

    # Draw valid contours on the black image
    cv2.drawContours(contour_image, valid_contours, -1, (255), thickness=2)

    return contour_image

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

    # Convert the NumPy array to a PIL Image
    edge_image = Image.fromarray((edge_array * 255).astype('uint8'))

    return edge_image

def apply_sobel(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=-1)

    # Apply Sobel edge detection
    edge_array = filters.sobel(image_array)

    # Normalize values to the range [0, 255]
    edge_array = (edge_array - np.min(edge_array)) / (np.max(edge_array) - np.min(edge_array)) * 255

    # Convert the NumPy array to a PIL Image
    edge_image = Image.fromarray(edge_array.astype('uint8'))

    return edge_image


def apply_kirsch(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=-1)

    # Apply Kirsch edge detection using a custom convolution kernel
    kirsch_kernel = np.array([
        [[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]],
        [[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]],
        [[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]],
        [[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]],
    ])

    # Convolve the image with each kernel
    kirsch_responses = [convolve2d(image_array, kernel, mode='same', boundary='symm') for kernel in kirsch_kernel]

    # Combine the responses into a single array
    kirsch_array = np.max(np.abs(kirsch_responses), axis=0)

    # Normalize values to the range [0, 255]
    kirsch_array = (kirsch_array - np.min(kirsch_array)) / (np.max(kirsch_array) - np.min(kirsch_array)) * 255

    # Convert the NumPy array to a PIL Image
    kirsch_image = Image.fromarray(kirsch_array.astype('uint8'))

    return kirsch_image

def apply_log(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=-1)

    # Apply LoG edge detection using scipy's gaussian_laplace
    log_array = gaussian_laplace(image_array, sigma=1)

    # Normalize values to the range [0, 255]
    log_array = (log_array - np.min(log_array)) / (np.max(log_array) - np.min(log_array)) * 255

    # Convert the NumPy array to a PIL Image
    log_image = Image.fromarray(log_array.astype('uint8'))

    return log_image

def apply_prewitt(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=-1)

    # Apply Prewitt edge detection using scipy's prewitt
    prewitt_array = filters.prewitt(image_array)

    # Normalize values to the range [0, 255]
    prewitt_array = (prewitt_array - np.min(prewitt_array)) / (np.max(prewitt_array) - np.min(prewitt_array)) * 255

    # Convert the NumPy array to a PIL Image
    prewitt_image = Image.fromarray(prewitt_array.astype('uint8'))

    return prewitt_image

def apply_hed(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    (H, W) = image.shape[:2]


    blob = cv2.dnn.blobFromImage(image_array, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward() 

    # Convert the edges to a uint8 image
    hed = (hed * 255).astype(np.uint8)

    hed_image = Image.fromarray(hed.squeeze(), mode='L')

    return hed_image
