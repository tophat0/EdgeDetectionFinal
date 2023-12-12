# edge_detection.py

from PIL import Image, ImageTk
from skimage import feature
import numpy as np

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
