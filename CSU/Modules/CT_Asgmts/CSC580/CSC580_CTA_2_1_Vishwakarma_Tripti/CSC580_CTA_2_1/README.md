# CSC580 CTA 2.1 – Classifying Handwritten Digits (MNIST)

This project trains a multi-layer perceptron (MLP) on the MNIST dataset using TensorFlow (TF1-style via `tf.compat.v1`). It includes scripts to run experiments, save misclassified images, and generate a Word report with findings and screenshots.

## Files
- `mnist_mlp_tf1.py` – Core training script; can be run directly or imported for programmatic experiments.
- `experiments.py` – Runs hyperparameter sweeps (hidden units, learning rates, batch sizes, layers) and saves plots.
- `generate_report.py` – Builds a Word document summarizing results, plots, and sample misclassifications.
- `requirements.txt` – Python dependencies.

## Quick Start
1. Create and activate a virtual environment (recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a single training (matches assignment baseline):
   ```bash
   python3 mnist_mlp_tf1.py --hidden_units 512 --learning_rate 0.5 --batch_size 100 --epochs 20 --layers 1 --save_miscl 20
   ```
   Outputs (accuracy and misclassified images) are saved in `outputs/`.

## Experiments (required by assignment)
Run a sweep to answer all questions:
```bash
python3 experiments.py
```
This saves:
- `outputs/experiments_results.json` – All runs with their test accuracy
- `outputs/acc_vs_hidden_units.png` – Accuracy vs hidden units
- `outputs/acc_vs_learning_rate.png` – Accuracy vs learning rate
- `outputs/acc_vs_batch_size.png` – Accuracy vs batch size
- `outputs/best_config.json` – Best accuracy configuration

You can adjust search spaces directly in `experiments.py`.

## Report Generation
After running experiments, generate the Word report:
```bash
python3 generate_report.py
```
This creates `CSC580_CTA_2_1_Report.docx` with:
- Best accuracy and configuration
- Plots showing hyperparameter effects
- Misclassified example images
- Code excerpts

## Answers to Prompts (how to obtain)
- Accuracy of the model: see terminal output or `outputs/last_run.json`.
- Misclassified images: see `outputs/misclassified/*.png`.
- Effect of more/fewer hidden neurons: run `experiments.py` and check `acc_vs_hidden_units.png`.
- Learning rates (≥4 values): results plotted in `acc_vs_learning_rate.png`.
- Adding another hidden layer: compare entries in `outputs/experiments_results.json` where `layers` is 1 vs 2.
- Batch sizes (≥3 values): see `acc_vs_batch_size.png`.
- Best accuracy: see `outputs/best_config.json`.

## Packaging for Submission
Create the zip named `CSC580_CTA_2_1_Vishwakarma_Tripti.zip` containing:
- `mnist_mlp_tf1.py`
- `experiments.py`
- `generate_report.py`
- `README.md`
- `requirements.txt`
- `CSC580_CTA_2_1_Report.docx` (after you generate it)
- Optional: `outputs/` with plots and sample misclassifications

## Notes
- The implementation follows the assignment’s TF1 placeholder/session pattern using `tf.compat.v1`.
- Default epochs in the sweep are set lower for speed; increase to 20 for final results.
- If TensorFlow wheels vary by platform, ensure you are using a compatible Python version.
