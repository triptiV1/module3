#!/usr/bin/env python3
"""
Face Detection Program for CSC580 (OpenCV Alternative)
Author: [Your Name]
Date: September 2025

This program detects faces in an image using OpenCV's Haar Cascade classifier
and draws red bounding boxes around detected faces.
"""

import cv2
import numpy as np
import sys
import os
from PIL import Image, ImageDraw

def detect_faces_in_image(image_path, output_path=None):
    """
    Detect faces in an image and draw red bounding boxes around them.
    
    Args:
        image_path (str): Path to the input image file
        output_path (str): Optional path for output image. If None, displays image.
    
    Returns:
        int: Number of faces detected
    """
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return 0
    
    try:
        # Load the image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image '{image_path}'. Please check the file format.")
            return 0
        
        # Convert to RGB for PIL compatibility
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Find all the faces in the image
        print("Detecting faces...")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert face locations to the format expected by the assignment
        face_locations = []
        for (x, y, w, h) in faces:
            # Convert from (x, y, width, height) to (top, right, bottom, left)
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))
        
        # Print number of faces found
        number_of_faces = len(face_locations)
        print("Found {} face(s) in this picture.".format(number_of_faces))
        
        if number_of_faces == 0:
            print("No faces detected in the image.")
            return 0
        
        # Load the image into a Python Image Library object so that you can draw on top of it and display it
        pil_image = Image.fromarray(rgb_image)
        
        # Create a drawing context
        draw_handle = ImageDraw.Draw(pil_image)
        
        # Process each detected face
        for i, face_location in enumerate(face_locations):
            # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            
            # Draw a box around the face
            draw_handle.rectangle([left, top, right, bottom], outline="red", width=3)
            
            # Optional: Add face number label
            draw_handle.text((left, top-20), f"Face {i+1}", fill="red")
        
        # Save or display the image
        if output_path:
            pil_image.save(output_path)
            print(f"Output image saved as: {output_path}")
        else:
            # Display the image on screen
            pil_image.show()
        
        return number_of_faces
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return 0

def main():
    """
    Main function to handle command line arguments and run face detection.
    """
    
    print("Face Detection Program for CSC580")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python face_detection_opencv.py <image_path> [output_path]")
        print("Example: python face_detection_opencv.py faces.png output_faces.png")
        return
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Detect faces in the image
    faces_detected = detect_faces_in_image(image_path, output_path)
    
    if faces_detected > 0:
        print(f"\nSuccess! Detected {faces_detected} face(s) in the image.")
        if output_path:
            print(f"Results saved to: {output_path}")
        else:
            print("Image displayed on screen.")
    else:
        print("\nNo faces were detected in the image.")

if __name__ == "__main__":
    main()
