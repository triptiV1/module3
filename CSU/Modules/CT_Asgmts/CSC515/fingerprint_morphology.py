import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# Use the full path provided by the user
image_path = '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/latent_fingerprint.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Define a kernel (structuring element)
    # A common kernel is a square matrix of ones. Size can be adjusted.
    kernel_size = 3 # You can change this size, e.g., 5, 7
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # --- Morphological Operations ---

    # 1. Erosion
    # Shrinks the bright areas and expands the dark areas.
    # You can change the number of iterations
    erosion = cv2.erode(img, kernel, iterations=1)

    # 2. Dilation
    # expands the bright areas and shrinks the dark areas.
    # You can change the number of iterations
    dilation = cv2.dilate(img, kernel, iterations=1)

    # 3. Opening
    # Erosion followed by Dilation. Useful for removing small objects (noise)
    # while preserving the shape and size of larger objects.
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 4. Closing
    # Dilation followed by Erosion. Useful for closing small holes inside
    # the foreground objects or small black points on the object.
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # --- Display Results ---
    # Using matplotlib to display multiple images easily
    plt.figure(figsize=(15, 10)) # Adjusted figure size for better viewing

    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4), plt.imshow(opening, cmap='gray'), plt.title('Opening (Erosion then Dilation)')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 5), plt.imshow(closing, cmap='gray'), plt.title('Closing (Dilation then Erosion)')
    plt.xticks([]), plt.yticks([])

    # Add an empty subplot or adjust layout if only 5 images are shown
    # plt.subplot(2, 3, 6).axis('off') # Optional: hide the 6th subplot

    plt.tight_layout() # Adjust layout to prevent titles overlapping
    plt.show()

    # --- Optional: Save the processed images ---
    # Uncomment the lines below if you want to save the output images
    # cv2.imwrite('erosion_result.png', erosion)
    # cv2.imwrite('dilation_result.png', dilation)
    # cv2.imwrite('opening_result.png', opening)
    # cv2.imwrite('closing_result.png', closing)
