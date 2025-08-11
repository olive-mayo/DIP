import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load grayscale image
img = cv2.imread('assets/food.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Step 1: Manual Histogram Computation
# -----------------------------
hist_manual = np.zeros(256, dtype=int)
for pixel in gray.flatten():
    hist_manual[pixel] += 1

# -----------------------------
# Step 2: OpenCV Histogram Equalization
# -----------------------------
equalized_cv = cv2.equalizeHist(gray)

# -----------------------------
# Step 3: Manual Histogram Equalization
# -----------------------------
cdf = hist_manual.cumsum()
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
cdf_normalized = cdf_normalized.astype(np.uint8)

equalized_manual = cdf_normalized[gray]

# -----------------------------
# Step 4: Plotting
# -----------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
ax = axes.ravel()

# Original image and histogram
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Original Grayscale')
ax[0].axis('off')

ax[1].bar(np.arange(256), hist_manual, color='black')
ax[1].set_title('Original Histogram (Manual)')

ax[2].hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
ax[2].set_title('Original Histogram (cv2 calc)')

# OpenCV Equalized
ax[3].imshow(equalized_cv, cmap='gray')
ax[3].set_title('Equalized (OpenCV)')
ax[3].axis('off')

ax[4].hist(equalized_cv.ravel(), bins=256, range=(0, 256), color='gray')
ax[4].set_title('Equalized Histogram (OpenCV)')

# Manual Equalized
ax[6].imshow(equalized_manual, cmap='gray')
ax[6].set_title('Equalized (Manual)')
ax[6].axis('off')

ax[5].axis('off')
ax[8].axis('off')

hist_eq_manual = np.zeros(256, dtype=int)
for pix in equalized_manual.flatten():
    hist_eq_manual[pix] += 1

ax[7].bar(np.arange(256), hist_eq_manual, color='black')
ax[7].set_title('Equalized Histogram (Manual)')

plt.tight_layout()
os.makedirs("outputs/hist_eq", exist_ok=True)
cv2.imwrite("outputs/hist_eq/equalized_cv.jpg", equalized_cv)
cv2.imwrite("outputs/hist_eq/equalized_manual.jpg", equalized_manual)
plt.savefig("outputs/hist_eq/histogram_comparison.png")
plt.show()
