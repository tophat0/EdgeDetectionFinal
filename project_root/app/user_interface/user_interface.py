import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from image_processing.edge_detection import edge_detection, apply_canny, prewitt_edge_detection, log_edge_detection, apply_hed, apply_sobel
from image_processing.image_isolation import isolate_and_count_objects, apply_edge_detection_functions

class ImageUploaderApp:
    def __init__(self, master):
        #Initialize UI popup
        self.master = master
        self.master.title("Edge Detection")
        self.master.geometry("1300x700")

        # Title above the image box
        self.title_label = tk.Label(self.master, text="Edge Detection")
        self.title_label.pack(pady=10)

        # Frame to organize the image and object count
        content_frame = tk.Frame(self.master)
        content_frame.pack(pady=10, padx=10, side=tk.LEFT)

        # Image label to display the edge-detected image
        self.image_label = tk.Label(content_frame)
        self.image_label.pack(pady=20)

        # Text above the "Upload Image" button
        self.upload_text_label = tk.Label(content_frame, text="Upload Image to be Analyzed")
        self.upload_text_label.pack(pady=10)

        # Upload button
        self.upload_button = tk.Button(content_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Label to display the number of objects
        self.object_count_label = tk.Label(content_frame, text="Number of Objects: 0")
        self.object_count_label.pack(pady=10)

        # Frame to organize the cutout images
        self.cutout_frame = tk.Frame(self.master)
        self.cutout_frame.pack(pady=10)


    # Displays the objects in the edge detected image
    def display_cutout_images(self, cutout_images):
        # Clear any previous content in the frame
        for widget in self.cutout_frame.winfo_children():
            widget.destroy()

        if cutout_images is None:
            # Display a message if no objects are detected
            message_label = tk.Label(self.cutout_frame, text="Objects not detected", font=("Helvetica", 12))
            message_label.grid(row=0, column=0, padx=10, pady=10)
        else:
            # Display cutout images in a 3x2 grid
            for i, cutout_image in enumerate(cutout_images):
                # Convert NumPy array to PhotoImage
                cutout_tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cutout_image, cv2.COLOR_BGR2RGB)))

                cutout_label = tk.Label(self.cutout_frame, image=cutout_tk_image)
                cutout_label.grid(row=i // 3, column=i % 3, padx=10, pady=10)

                # Keep a reference to the image to prevent it from being garbage collected
                cutout_label.image = cutout_tk_image

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path, max_width=300, max_height=300):
        # Read the image using OpenCV
        original_image = cv2.imread(file_path)

        # Get the height, width, and aspect ratio of the image
        width, height = original_image.shape[1], original_image.shape[0]
        aspect_ratio = width / height

        # Resize the image while maintaining aspect ratio
        if aspect_ratio > 1:  # Landscape image
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait image
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image using OpenCV
        resized_image = cv2.resize(original_image, (new_width, new_height))

        # Apply our custom edge detection
        edge_image, objects = edge_detection(resized_image)

        # Convert the edge detected image to a PIL Image and create PhotoImage
        self.uploaded_image = ImageTk.PhotoImage(Image.fromarray(np.array(edge_image)))

        # Get objects from the contour edge-detected image
        total_objects, num_objects, cutout_images = isolate_and_count_objects(np.array(objects))

        # Display the edge-detected image and update the object count label
        display_text = f"Number of Objects: {total_objects}"
        display_text = f"Five largest objects -->"
        self.image_label.config(image=self.uploaded_image)
        self.image_label.image = self.uploaded_image

        self.object_count_label.config(text=display_text)

        # Display the cutout images
        self.display_cutout_images(cutout_images)

        # Open the results of all the edge detection methods in a new window
        edgeimageslist = apply_edge_detection_functions(file_path)
        self.display_cutout_images(edgeimageslist)

def run_app():
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
