import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the noisy image using the specified path
image = cv2.imread('/Users/tvishwak/Documents/GitHub/module3/CSU/Modules/CT_Asgmts/CSC515/Impulse_noise_image.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Define kernel sizes and Gaussian sigmas
kernels = [3, 5, 7]
sigma_values = [1, 2]

# Step 3: Initialize a figure for subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
fig.suptitle('Comparison of Filters with Different Kernels and Gaussian Sigmas', fontsize=16)

# Step 4: Apply filters and store results
for i, kernel_size in enumerate(kernels):
    # Mean filter
    mean_filtered = cv2.blur(image, (kernel_size, kernel_size))
    
    # Median filter
    median_filtered = cv2.medianBlur(image, kernel_size)
    
    # Gaussian filter with sigma 1
    gaussian_filtered_sigma1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_values[0])
    
    # Gaussian filter with sigma 2
    gaussian_filtered_sigma2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_values[1])
    
    # Plot results
    axes[i, 0].imshow(mean_filtered, cmap='gray')
    axes[i, 0].set_title(f'Mean Filter ({kernel_size}x{kernel_size})')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(median_filtered, cmap='gray')
    axes[i, 1].set_title(f'Median Filter ({kernel_size}x{kernel_size})')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(gaussian_filtered_sigma1, cmap='gray')
    axes[i, 2].set_title(f'Gaussian Filter ({kernel_size}x{kernel_size}, σ={sigma_values[0]})')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(gaussian_filtered_sigma2, cmap='gray')
    axes[i, 3].set_title(f'Gaussian Filter ({kernel_size}x{kernel_size}, σ={sigma_values[1]})')
    axes[i, 3].axis('off')

# Add row and column labels
axes[0, 0].set_ylabel('3x3 Kernel', fontsize=12)
axes[1, 0].set_ylabel('5x5 Kernel', fontsize=12)
axes[2, 0].set_ylabel('7x7 Kernel', fontsize=12)

for ax in axes[-1, :]:
    ax.set_xlabel('Filter Type', fontsize=12)

# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('filter_comparison.png')
plt.show()
