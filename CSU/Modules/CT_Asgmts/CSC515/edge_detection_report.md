# Edge Detection Performance Analysis: Canny vs. Sobel vs. Laplacian

This report details an empirical evaluation of three common edge detection algorithms—Canny, Sobel, and Laplacian—on a synthetic image. The goal is to rigorously assess their performance under ideal, noisy, and low-contrast conditions.

## 1. Evaluation Methodology

To quantitatively evaluate the performance of each edge detector, a pixel-based comparison between the detected edge map and a known ground truth was performed. The ground truth precisely outlines the shapes in the synthetic image.

### My Evaluation Metric: F1-Score

The primary metric used is the **F1-Score**, which is the harmonic mean of Precision and Recall. These components are defined as:

-   **True Positives (TP):** The number of pixels correctly identified as edges.
-   **False Positives (FP):** The number of pixels incorrectly identified as edges (i.e., detecting an edge where none exists). These represent noise or artifacts.
-   **False Negatives (FN):** The number of actual edge pixels that the detector failed to identify. These represent missed edges or gaps.

From these, we calculate:

-   **Precision** = `TP / (TP + FP)`
    -   *Measures the accuracy of the detected edges. A high precision means fewer false edges.*
-   **Recall** = `TP / (TP + FN)`
    -   *Measures the completeness of the detected edges. A high recall means fewer missed edges.*
-   **F1-Score** = `2 * (Precision * Recall) / (Precision + Recall)`
    -   *Provides a single, balanced measure of performance. It is high only when both precision and recall are high, making it ideal for evaluating edge detectors where there is a trade-off between detecting too many spurious edges and missing genuine ones.*

This method was chosen because it directly addresses the two main failure modes of an edge detector: spurious responses and missed detections. It is simple to implement and provides a clear, interpretable score.

### Comparison to Other Evaluation Methods

1.  **Pratt's Figure of Merit (FOM):**
    -   **Description:** Pratt's FOM is a classic and highly regarded evaluation metric. It not only considers missed and false edges but also penalizes detected edge pixels based on their distance from the true edge location. It rewards continuity and localization accuracy.
    -   **Comparison:** FOM is more sophisticated than my F1-score method as it accounts for the *localization* of edges, not just their detection. An edge detected one pixel away from the ground truth is penalized less than one detected five pixels away. While superior in capturing localization quality, it is more complex to implement. For this application, where the ground truth is perfectly defined, the F1-score provides a sufficient and clear measure of detection success.

2.  **Mean Squared Error (MSE) / Peak Signal-to-Noise Ratio (PSNR):**
    -   **Description:** These are general-purpose image fidelity metrics that measure the average squared difference between pixel values of two images. PSNR is derived from MSE.
    -   **Comparison:** MSE and PSNR are poorly suited for evaluating edge detection. They measure per-pixel intensity differences, not structural correctness. A faint, noisy edge map (FP) might have a better MSE score than a clean but slightly displaced edge map, even though the latter is structurally superior. The F1-score, by contrast, operates on a binary classification of pixels (edge vs. non-edge), which is far more relevant to the task's objective.

## 2. Performance Analysis

The accompanying Python script (`edge_detection_evaluation.py`) was executed to generate the following metrics.

### Condition 1: Clean Image

-   **Canny**: `F1-Score=0.881` (Precision=0.978, Recall=0.801)
-   **Sobel**: `F1-Score=0.741` (Precision=0.608, Recall=0.947)
-   **Laplacian**: `F1-Score=0.219` (Precision=0.124, Recall=0.963)

**Analysis:**
-   **Canny** achieves the highest F1-Score. Its high precision indicates very few false edges, a result of its non-maximum suppression and hysteresis thresholding. The recall is strong but not perfect, as the single-pixel-thin edges of the ground truth are hard to match perfectly.
-   **Sobel** has excellent recall, detecting most of the true edges. However, its precision is low because it produces thick, smeared edges, leading to a high number of false positives around the true edge line.
-   **Laplacian** performs poorly. While it detects almost all edges (very high recall), it produces double edges and is extremely sensitive, resulting in a massive number of false positives and a dismal precision score.

### Condition 2: Noisy Image (Gaussian Noise, sigma=25)

-   **Canny**: `F1-Score=0.825` (Precision=0.808, Recall=0.842)
-   **Sobel**: `F1-Score=0.231` (Precision=0.133, Recall=0.884)
-   **Laplacian**: `F1-Score=0.103` (Precision=0.055, Recall=0.882)

**Analysis:**
-   **Canny** demonstrates remarkable robustness to noise. Its F1-score only drops slightly. This is because the initial Gaussian smoothing step in the Canny algorithm is specifically designed to suppress such noise before differentiation.
-   **Sobel**'s performance collapses. The noise introduces many high-frequency gradients, which the Sobel operator falsely detects as edges, causing precision to plummet.
-   **Laplacian**, being a second-derivative operator, is even more susceptible to noise. Its performance degrades significantly, rendering it practically useless in this scenario.

### Condition 3: Low Contrast Image

-   **Canny**: `F1-Score=0.871` (Precision=0.978, Recall=0.785)
-   **Sobel**: `F1-Score=0.718` (Precision=0.581, Recall=0.941)
-   **Laplacian**: `F1-Score=0.198` (Precision=0.111, Recall=0.942)

**Analysis:**
-   **Canny** continues to perform exceptionally well. The adaptive nature of its hysteresis thresholding helps it correctly identify the edges despite the lower intensity difference.
-   **Sobel** and **Laplacian** see a slight performance drop but are less affected by contrast reduction than by noise. Their reliance on a fixed threshold means that as long as the gradient is above the threshold, the edge is detected. However, their fundamental flaws (thick/double edges) remain.

## 3. Conclusion

Based on the quantitative evaluation using the F1-score, the **Canny edge detector is unequivocally the best-performing method** across all tested conditions.

-   **Under ideal conditions**, it provides the best balance of precision and recall, yielding clean, single-pixel-thin edges.
-   **Under noisy conditions**, its superiority is most evident. The built-in noise reduction makes it highly robust, whereas derivative-based methods like Sobel and Laplacian fail dramatically.
-   **Under low-contrast conditions**, it remains reliable and significantly outperforms the other two methods.

The Sobel operator serves as a simple and fast way to get a basic gradient magnitude image but is not a sophisticated edge detector. The Laplacian operator is highly sensitive to noise and is generally better suited for finding fine details or as part of other algorithms (e.g., Laplacian of Gaussian) rather than as a standalone edge detector. This experiment substantiates why the Canny algorithm is the standard and most widely recommended method for general-purpose edge detection.
