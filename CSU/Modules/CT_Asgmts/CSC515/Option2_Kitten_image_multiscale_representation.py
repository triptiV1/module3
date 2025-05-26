import cv2
import sys

# Path to the input image
image_path = '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/Kitty_image.jpg'

# Step 1: Display the original image and extract the color channels
def extract_channels(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
        sys.exit(1)  # Exit with an error code

    # Display the original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(3000)  # Wait for 3 seconds before automatically closing
    cv2.destroyAllWindows()

    # Split the image into its three color channels (Blue, Green, Red)
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Save the individual channels as grayscale images
    cv2.imwrite('/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/blue_channel.jpg', blue_channel)
    cv2.imwrite('/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/green_channel.jpg', green_channel)
    cv2.imwrite('/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/red_channel.jpg', red_channel)
    print("Channels saved successfully!")

    # Display the individual channels
    cv2.imshow('Blue Channel', blue_channel)
    cv2.imshow('Green Channel', green_channel)
    cv2.imshow('Red Channel', red_channel)
    cv2.waitKey(3000)  # Wait for 3 seconds before closing
    cv2.destroyAllWindows()

    return blue_channel, green_channel, red_channel

# Step 2: Merge the channels back into a single colored image
def merge_channels(blue_channel, green_channel, red_channel):
    # Merge the channels back into a colored image
    merged_image = cv2.merge([blue_channel, green_channel, red_channel])

    # Save the merged image
    merged_image_path = '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/merged_image.jpg'
    cv2.imwrite(merged_image_path, merged_image)
    print(f"Merged image saved at: {merged_image_path}")

    # Display the merged image
    cv2.imshow('Merged Image (Original)', merged_image)
    cv2.waitKey(3000)  # Wait for 3 seconds before closing
    cv2.destroyAllWindows()

    return merged_image

# Step 3: Swap the red and green channels, and merge them into a GRB image
def swap_channels(blue_channel, green_channel, red_channel):
    # Merge the channels in the order Green, Red, Blue (GRB)
    swapped_image = cv2.merge([green_channel, red_channel, blue_channel])

    # Save the swapped image
    swapped_image_path = '/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/swapped_grb_image.jpg'
    cv2.imwrite(swapped_image_path, swapped_image)
    print(f"Swapped (GRB) image saved at: {swapped_image_path}")

    # Display the swapped image
    cv2.imshow('Swapped Channels Image (GRB)', swapped_image)
    cv2.waitKey(3000)  # Wait for 3 seconds before closing
    cv2.destroyAllWindows()

    return swapped_image

# Main function to execute the steps
def main():
    # Step 1: Display the original image and extract the channels
    blue_channel, green_channel, red_channel = extract_channels(image_path)

    # Step 2: Merge the channels back into a colored 3D image
    merged_image = merge_channels(blue_channel, green_channel, red_channel)

    # Step 3: Swap the red and green channels, and merge them back (GRB format)
    swapped_image = swap_channels(blue_channel, green_channel, red_channel)

    print("Processing completed successfully! Exiting program.")
    sys.exit(0)  # Explicitly exit the program

# Run the main function
if __name__ == "__main__":
    main()
