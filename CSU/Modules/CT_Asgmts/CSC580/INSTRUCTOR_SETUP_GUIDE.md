# CSC580 Face Detection - Setup Instructions for Instructor

## Student: Tripti Vishwakarma
## Assignment: CSC580_CTA_1_1 Face Detection

---

## How to Run the Submission

### Step 1: Extract the Zip File
```bash
unzip CSC580_CTA_1_1_Vishwakarma_Tripti.zip
cd extracted_folder/
```

### Step 2: Install Dependencies
```bash
pip install opencv-python pillow numpy
```

### Step 3: Run the Program
```bash
python face_detection.py faces.png output.png
```

---

## Expected Output
```
Face Detection Program for CSC580
========================================
Loading image: faces.png
Detecting faces...
Found 3 face(s) in this picture.
A face is located at pixel location Top: 190, Left: 60, Bottom: 231, Right: 101
A face is located at pixel location Top: 188, Left: 591, Bottom: 235, Right: 638
A face is located at pixel location Top: 195, Left: 235, Bottom: 242, Right: 282
Output image saved as: output.png

Success! Detected 3 face(s) in the image.
Results saved to: output.png
```

---

## Files Included in Submission
- `face_detection.py` - Main Python program
- `faces.png` - Input image (3 faces)
- `faces_detected_original.png` - Sample output with red bounding boxes
- `requirements.txt` - Dependencies list

---

## Program Features
- Detects faces using face_recognition library (with OpenCV fallback)
- Draws red bounding boxes around detected faces
- Prints face coordinates in (top, left, bottom, right) format
- Uses PIL.ImageDraw as specified in assignment
- Follows provided starter code structure

---

## Alternative: Using Virtual Environment
```bash
python3 -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
python face_detection.py faces.png output.png
deactivate
```

---

## Troubleshooting
- If face_recognition installation fails, program automatically uses OpenCV fallback
- Ensure Python 3.7+ is installed
- All dependencies are listed in requirements.txt
