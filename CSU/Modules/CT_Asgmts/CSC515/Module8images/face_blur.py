import cv2
import numpy as np

def blur_eyes(image_path, face_cascade_path, eye_cascade_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascades
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces and blur eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            # Apply Gaussian blur
            blurred_eye = cv2.GaussianBlur(eye_roi, (23, 23), 30)
            roi_color[ey:ey+eh, ex:ex+ew] = blurred_eye

    # Save the output image
    output_path = image_path.split('.')[0] + '_processed.png'
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    eye_cascade_path = 'haarcascade_eye.xml'

    images = ['womenImage.png', 'manFarAway.png', 'dog.png']

    for image in images:
        blur_eyes(image, face_cascade_path, eye_cascade_path)
