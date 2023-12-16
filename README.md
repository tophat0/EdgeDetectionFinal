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
**main.py** : Window will pop up with an option to select an image file. Once image file is selected, the window will display the edge detection result as well as the 5 largest objects in that image. Another window will also pop up showing the edge detection results of the other algorithms.
**evaluation.py**: Graph of the runtimes of the edge detection algorithms,  the edge detection results of the algorithms, the result of our edge detection algorithm saved to output folder
