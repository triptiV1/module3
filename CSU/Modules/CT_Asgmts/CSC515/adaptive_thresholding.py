import cv2
import matplotlib.pyplot as plt

def segment_image(image_path, block_size=11, C=2):
    """
    Loads an image, applies adaptive thresholding, and displays the result.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, C)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

# Image paths
image_paths = [
    '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/images/indoor.jpeg',
    '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/images/outdoor.jpeg',
    '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/images/closeup.jpeg'
]

# Process each image
for path in image_paths:
    segment_image(path)
