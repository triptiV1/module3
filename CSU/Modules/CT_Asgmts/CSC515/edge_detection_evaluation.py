import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_synthetic_image(width=512, height=512, bg_intensity=75, shape_intensity=200):
    """
    Generates a synthetic image with a square and a circle, and its ground truth edge map.
    """
    # Create a blank image
    image = np.full((height, width), bg_intensity, dtype=np.uint8)
    ground_truth = np.zeros((height, width), dtype=np.uint8)

    # Define shape properties
    sq_top_left = (80, 80)
    sq_bottom_right = (220, 220)
    circle_center = (350, 350)
    circle_radius = 80

    # Draw filled shapes on the main image
    cv2.rectangle(image, sq_top_left, sq_bottom_right, shape_intensity, -1)
    cv2.circle(image, circle_center, circle_radius, shape_intensity, -1)

    # Draw outlines for the ground truth
    cv2.rectangle(ground_truth, sq_top_left, sq_bottom_right, 255, 1)
    cv2.circle(ground_truth, circle_center, circle_radius, 255, 1)

    return image, ground_truth

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to the image."""
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def evaluate_performance(ground_truth, detected_edges):
    """
    Calculates Precision, Recall, and F1-Score for edge detection performance.
    """
    # Ensure both images are binary (0 or 255)
    ground_truth_bin = ground_truth > 0
    detected_edges_bin = detected_edges > 0

    tp = np.sum(np.logical_and(detected_edges_bin, ground_truth_bin))
    fp = np.sum(np.logical_and(detected_edges_bin, np.logical_not(ground_truth_bin)))
    fn = np.sum(np.logical_and(np.logical_not(detected_edges_bin), ground_truth_bin))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def run_and_evaluate(image, ground_truth, condition_name):
    """Runs all edge detectors on an image and prints their performance."""
    print(f"--- Evaluation for: {condition_name} ---")

    # 1. Canny Edge Detector
    canny_edges = cv2.Canny(image, 50, 150)
    p_c, r_c, f1_c = evaluate_performance(ground_truth, canny_edges)
    print(f"Canny    : Precision={p_c:.3f}, Recall={r_c:.3f}, F1-Score={f1_c:.3f}")

    # 2. Sobel Edge Detector
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    _, sobel_edges = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    p_s, r_s, f1_s = evaluate_performance(ground_truth, sobel_edges)
    print(f"Sobel    : Precision={p_s:.3f}, Recall={r_s:.3f}, F1-Score={f1_s:.3f}")

    # 3. Laplacian Edge Detector
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    _, laplacian_edges = cv2.threshold(laplacian_abs, 50, 255, cv2.THRESH_BINARY)
    p_l, r_l, f1_l = evaluate_performance(ground_truth, laplacian_edges)
    print(f"Laplacian: Precision={p_l:.3f}, Recall={r_l:.3f}, F1-Score={f1_l:.3f}")
    print("-" * 40)

    return canny_edges, sobel_edges, laplacian_edges

def main():
    # --- Experiment 1: Ideal Conditions ---
    image_clean, ground_truth = create_synthetic_image()
    canny_clean, sobel_clean, lap_clean = run_and_evaluate(image_clean, ground_truth, "Clean Image")

    # --- Experiment 2: Noisy Conditions ---
    image_noisy = add_gaussian_noise(image_clean, sigma=25)
    canny_noisy, sobel_noisy, lap_noisy = run_and_evaluate(image_noisy, ground_truth, "Noisy Image (Sigma=25)")

    # --- Experiment 3: Low Contrast Conditions ---
    image_low_contrast, _ = create_synthetic_image(bg_intensity=100, shape_intensity=150)
    canny_low, sobel_low, lap_low = run_and_evaluate(image_low_contrast, ground_truth, "Low Contrast Image")

    # --- Visualization ---
    titles = [
        'Clean Image', 'Ground Truth', 'Canny', 'Sobel', 'Laplacian',
        'Noisy Image', 'Canny (Noisy)', 'Sobel (Noisy)', 'Laplacian (Noisy)',
        'Low Contrast', 'Canny (Low)', 'Sobel (Low)', 'Laplacian (Low)'
    ]
    images = [
        image_clean, ground_truth, canny_clean, sobel_clean, lap_clean,
        image_noisy, canny_noisy, sobel_noisy, lap_noisy,
        image_low_contrast, canny_low, sobel_low, lap_low
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        if i < len(titles):
            plt.subplot(3, 5, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
