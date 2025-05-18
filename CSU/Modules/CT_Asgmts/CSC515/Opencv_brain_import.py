import cv2

# Path to the original image
image_path = "/Users/tvishwak/Downloads/brain.jpg"

# Path to save the copied image
save_path = "/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/brain_copy.jpg"

try:
    # Step 1: Import the image
    brain_image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if brain_image is None:
        raise FileNotFoundError(f"The file '{image_path}' was not found or could not be loaded.")

    # Step 2: Display the image
    cv2.imshow("Brain Image", brain_image)  # Display the image in a window
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the image window

    # Step 3: Write a copy of the image to the specified directory
    success = cv2.imwrite(save_path, brain_image)

    if success:
        print(f"Image successfully saved to: {save_path}")
    else:
        print("Failed to save the image.")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
