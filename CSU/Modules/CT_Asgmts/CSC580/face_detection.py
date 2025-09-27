#!/usr/bin/env python3
"""
Face Detection Program for CSC580
Author: [Your Name]
Date: September 2025

This program detects faces in an image using the face_recognition library
and draws red bounding boxes around detected faces.
"""

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: face_recognition library not available. Using OpenCV fallback.")
    import cv2
    FACE_RECOGNITION_AVAILABLE = False

import PIL.Image
import PIL.ImageDraw
import numpy as np
import sys
import os

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
        # Load the jpg file into a numpy array
        print(f"Loading image: {image_path}")
        
        if FACE_RECOGNITION_AVAILABLE:
            image = face_recognition.load_image_file(image_path)
            # Find all the faces in the image
            print("Detecting faces...")
            face_locations = face_recognition.face_locations(image)
        else:
            # Fallback to OpenCV if face_recognition is not available
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                print(f"Error: Could not load image '{image_path}'.")
                return 0
            
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Detecting faces...")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert to face_recognition format (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
        
        # Print number of faces found
        number_of_faces = len(face_locations)
        print("Found {} face(s) in this picture.".format(number_of_faces))
        
        if number_of_faces == 0:
            print("No faces detected in the image.")
            return 0
        
        # Load the image into a Python Image Library object so that you can draw on top of it and display it
        pil_image = PIL.Image.fromarray(image)
        
        # Create a drawing context
        draw_handle = PIL.ImageDraw.Draw(pil_image)
        
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
        print("Usage: python face_detection.py <image_path> [output_path]")
        print("Example: python face_detection.py faces.png output_faces.png")
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
