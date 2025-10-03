# CSC580 CTA 3.2 – Predicting Fuel Efficiency Using TensorFlow (Auto MPG)

Author: Tripti Vishwakarma  
Date: <Insert Date>

---

## 1. Introduction
Fuel efficiency (MPG) prediction is a classic regression task. In this project, I used the UCI Auto MPG dataset (model years 1970–1982) and built TensorFlow/Keras models to predict MPG from vehicle attributes. Features include Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, and one-hot encoded Origin (USA, Europe, Japan). The target variable is MPG.

<Write 1–2 short paragraphs summarizing objectives and approach.>

---

## 2. Data Preparation
The dataset was cleaned by dropping rows with missing values. The `Origin` categorical field was one-hot encoded into `USA`, `Europe`, and `Japan` columns.

- **Figure 1: Dataset tail (terminal screenshot)**  
<Insert Figure 1 here>

The data was split into 80% training and 20% test for evaluation.

- **Figure 2: Train/test sizes line (terminal screenshot)**  
<Insert Figure 2 here>

---

## 3. Exploratory Analysis
A pairplot was generated to visualize relationships between features and MPG.

- **Figure 3: Pairplot (screenshot of outputs/pairplot.png)**  
<Insert Figure 3 here>

Key observations:
- Higher `Weight`, `Displacement`, and `Cylinders` correlate with lower `MPG`.
- `Model Year` shows improved efficiency over time.

---

## 4. Normalization
Z-score normalization was applied using the training set’s means and standard deviations to stabilize optimization and ensure comparable feature scales.

- **Figure 4: Train stats tail (terminal screenshot)**  
<Insert Figure 4 here>

---

## 5. Model Architecture
Two identical network architectures were trained with different losses:
- Architecture: Dense(64, ReLU) → Dense(64, ReLU) → Dense(1), optimizer RMSprop(0.001)
- Loss variants: Mean Squared Error (MSE) and Mean Absolute Error (MAE)

- **Figure 5: Model summary – MSE-loss model (terminal screenshot)**  
<Insert Figure 5 here>

- **Figure 6: Model summary – MAE-loss model (terminal screenshot)**  
<Insert Figure 6 here>

Rationale:
- MSE penalizes larger errors more strongly.
- MAE is more robust to outliers and often yields lower MAE scores.

---

## 6. Training Results
Training and validation curves were recorded for both losses.

- **Figure 7: MAE curve (screenshot of outputs/history_mae.png)**  
<Insert Figure 7 here>

- **Figure 8: MSE curve (screenshot of outputs/history_mse.png)**  
<Insert Figure 8 here>

Comments on convergence:
- Rapid improvement in early epochs followed by gradual convergence.
- A small gap between training and validation curves is typical; persistent larger gaps may indicate mild overfitting.

---

## 7. Metrics and History
The last five epochs from the training histories are shown to summarize end-of-training metrics.

- **Figure 9: History tail – MSE-loss model (terminal screenshot)**  
<Insert Figure 9 here>

- **Figure 10: History tail – MAE-loss model (terminal screenshot)**  
<Insert Figure 10 here>

Observations:
- Metrics stabilize near the end of training.
- Validation metrics plateau, indicating adequate training progress.

---

## 8. Test Set Evaluation
Final evaluations on the held-out test set:

- **Figure 11: Test-set evaluation lines (terminal screenshot)**  
<Insert Figure 11 here>

Interpretation:
- MSE-loss model emphasizes penalizing larger errors; often better MSE.
- MAE-loss model is robust to outliers; often slightly lower MAE.

<Insert your concrete numbers here from your run and briefly interpret which model behaved better and why.>

---

## 9. Conclusions
- Key drivers of MPG: `Weight`, `Displacement`, and `Cylinders` negatively correlate with MPG; `Model Year` positively correlates.
- Preferred loss (choose one): <MSE or MAE>, based on observed metrics and assignment guidance.
- Future improvements: early stopping, L2 regularization or dropout, learning-rate tuning, deeper/shallower networks, and feature engineering (e.g., power-to-weight, interactions).

---

## Appendix
All generated artifacts are saved under `CSU/Modules/CT_Asgmts/CSC580/CSC580_CTA_3_2/outputs/`:
- Pairplot: `pairplot.png`
- Training curves: `history_mae.png`, `history_mse.png`
- Text artifacts: `dataset_tail.txt`, `train_stats_tail.txt`, `model_summary_mse.txt`, `model_summary_mae.txt`, `history_tail_mse.txt`, `history_tail_mae.txt`

<Add any additional notes or references here.>
## References

Dua, D., & Graff, C. (2017). UCI Machine Learning Repository: Auto MPG Data Set. University of California, Irvine. https://archive.ics.uci.edu/dataset/9/auto+mpg

TensorFlow. (n.d.). Regression: Predict fuel efficiency. https://www.tensorflow.org/tutorials/keras/regression

Keras Team. (n.d.). Keras API reference. https://keras.io/api/

Tieleman, T., & Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning. https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

Waskom, M. L. (2021). Seaborn: Statistical data visualization. PairGrid and pairplot. https://seaborn.pydata.org/generated/seaborn.pairplot.html

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. Climate Research, 30(1), 79–82. https://doi.org/10.3354/cr030079

Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer.