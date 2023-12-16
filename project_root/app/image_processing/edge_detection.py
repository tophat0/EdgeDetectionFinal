# edge_detection.py

from PIL import Image
from skimage import feature, filters
import numpy as np
import cv2
from scipy.signal import convolve2d

# Import Pre-Trained HED Model
net = cv2.dnn.readNetFromCaffe('app/image_processing/deploy.prototxt', 'app/image_processing/hed_pretrained_bsds.caffemodel')

# Our edge detection method
def edge_detection(image):
    k_size=3           # Gaussian kernel size
    threshold=75       # Edge detection threshold
    min_area=3         # Object boundary threshold

    # Step 1: Smooth the image
    blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)

    # Step 2: Compute dark channel prior
    dark_channel = np.min(blurred, axis=2)
    dark_channel_percentile = np.percentile(dark_channel, 0.1)

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

    return edges, contour_image


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

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Prewitt operators for horizontal and vertical gradients
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    gradient_x = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_kernel_x)
    gradient_y = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_kernel_y)

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

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

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

    # Apply Kirsch edge detection using a convolution kernel
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

def apply_hed(image):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Get the height and width of the image
    (H, W) = image_array.shape[:2]

    # Puts image into HED processinf format, use the model to find edges
    blob = cv2.dnn.blobFromImage(image_array, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward() 

    # Convert the edges to a uint8 image
    hed = (hed * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image in grayscale mode
    hed_image = Image.fromarray(hed.squeeze(), mode='L')

    return hed_image
