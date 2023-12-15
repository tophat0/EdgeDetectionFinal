import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from app.image_processing.edge_detection import apply_canny, prewitt_edge_detection, log_edge_detection, sobel_edges, edge_detection
from app.image_processing.image_isolation import isolate_and_count_objects, apply_edge_detection_functions

class ImageUploaderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Edge Detection")
        self.master.geometry("1200x700")

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

        # New label to display the number of objects
        self.object_count_label = tk.Label(content_frame, text="Number of Objects: 0")
        self.object_count_label.pack(pady=10)

        # Frame to organize the cutout images in a 2x2 grid
        self.cutout_frame = tk.Frame(self.master)
        self.cutout_frame.pack(pady=10)

        # Blank box titled 'Five Largest Objects'
        self.blank_box_frame = tk.Frame(self.master)
        self.blank_box_frame.pack(pady=10, padx=10, side=tk.RIGHT, anchor=tk.N)
        self.display_blank_box(0)

    def display_blank_box(self, total_objects):
        # Clear any previous content in the frame
        for widget in self.blank_box_frame.winfo_children():
            widget.destroy()

        # Add a title to the blank box
        title_label = tk.Label(self.blank_box_frame, text="Five Largest Objects")
        title_label.pack(pady=10)

        # Add the total number of objects text
        total_objects_label = tk.Label(self.blank_box_frame, text=f"Total Objects: {total_objects}")
        total_objects_label.pack(pady=5)

        # Add a blank box (you can customize the size as needed)
        blank_box = tk.Label(self.blank_box_frame, text="", width=30, height=10, borderwidth=1, relief="solid")
        blank_box.pack(pady=10)

    def display_cutout_images(self, cutout_images):
        # Clear any previous content in the frame
        for widget in self.cutout_frame.winfo_children():
            widget.destroy()

        # Display cutout images in a 2x2 grid
        for i, cutout_image in enumerate(cutout_images):
            # Convert NumPy array to PhotoImage
            cutout_tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cutout_image, cv2.COLOR_BGR2RGB)))

            cutout_label = tk.Label(self.cutout_frame, image=cutout_tk_image)
            cutout_label.grid(row=i // 2, column=i % 2, padx=10, pady=10)

            # Keep a reference to the image to prevent it from being garbage collected
            cutout_label.image = cutout_tk_image

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path, max_width=300, max_height=300):
        # Read the image using OpenCV
        original_image = cv2.imread(file_path)

        # Resize the image while maintaining aspect ratio
        width, height = original_image.shape[1], original_image.shape[0]
        aspect_ratio = width / height

        if aspect_ratio > 1:  # Landscape image
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait image
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image using OpenCV
        resized_image = cv2.resize(original_image, (new_width, new_height))
        # Apply Canny edge detection
        edge_image = edge_detection(resized_image)

        # Convert the NumPy array to a PIL Image and create PhotoImage
        self.uploaded_image = ImageTk.PhotoImage(Image.fromarray(np.array(edge_image)))

        # Isolate and count objects directly from the edge-detected image
        total_objects, num_objects, cutout_images = isolate_and_count_objects(np.array(edge_image))

        # Display the edge-detected image and update the object count label
        display_text = f"Number of Objects: {num_objects}"
        self.image_label.config(image=self.uploaded_image)
        self.image_label.image = self.uploaded_image

        self.object_count_label.config(text=display_text)

        # Display the blank box and cutout images
        self.display_blank_box(total_objects=total_objects)
        self.display_cutout_images(cutout_images)

        edgeimageslist = apply_edge_detection_functions(file_path)

        self.display_cutout_images(edgeimageslist)

def run_app():
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
