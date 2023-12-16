# EdgeDetectionFinal

**By: Peyton Slater, Eric Gomez**

# How to Run

1. Download the zip file and open it in a Python editor or terminal.
2. Use the terminal to navigate to the project_root directory.
3. Run 'main.py' for the program

# Output Folder

The "output" folder contains images from the Berkeley Segmentation Dataset 500 that have been processed using our edge detection algorithm. 

## Edge Detection Algorithms

1. **Edge Detection**: Our implemented edge detection method.
2. **Canny Edge Detection**: Image processed using the Canny edge detection algorithm.
3. **Prewitt Edge Detection**: Image processed using the Prewitt edge detection operators.
4. **Log Edge Detection**: Image processed using the Laplacian of Gaussian (LoG) edge detection.
5. **Kirsch Edge Detection**: Image processed using the Kirsch edge detection.
6. **Sobel Edges**: Image processed using the Sobel operators for horizontal and vertical gradients.
7. **Hed Detection**: Image processed using the HED (Holistically-Nested Edge Detection) algorithm.

Feel free to explore and compare the results of each algorithm on the input images.

## Expected Outputs

### main.py

1. **Window Display:**
   - Upon running `main.py`, a window will pop up with an option to select an image file.
   - After selecting an image file, the window will display:
     - The result of ou redge detection algorithm on the image
     - The 5 largest objects in the image.
![image](https://github.com/tophat0/EdgeDetectionFinal/assets/69655459/04c0ac2f-ceed-4595-b30e-006cef6227bb)

2. **Additional Window:**
   - Another window will also pop up showing the edge detection results of the other algorithms.
![image](https://github.com/tophat0/EdgeDetectionFinal/assets/69655459/462f9997-fcc3-48d7-a464-84bd4a655806)

### evaluation.py

1. **Graph:**
   - A graph showing the runtimes of the edge detection algorithms.

2. **Algorithm Results:**
   - The edge detection results of all the algorithms will be displayed.

3. **Saving Output:**
   - The result of our edge detection algorithm on BDSD500 will be saved to the output folder.
