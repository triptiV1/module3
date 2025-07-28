import cv2

# Load Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the image
image_path = "/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/Me.jpeg"
image = cv2.imread(image_path)

# Ensure the image is loaded
if image is None:
    print("Error: Image not found at the specified path")
    exit()

# Convert to grayscale for better detection accuracy
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# If a face is detected, annotate it
for (x, y, w, h) in faces:
    # Draw a green circle around the face
    center = (x + w//2, y + h//2)
    radius = max(w, h) // 2
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    # Detect eyes within the detected face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Draw red bounding boxes around eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# Add text to the image
cv2.putText(image, "This is me", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Display the annotated image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated image
output_path = "/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/Annotated_Me.jpeg"
cv2.imwrite(output_path, image)
