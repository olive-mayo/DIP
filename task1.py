import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os

os.makedirs("outputs", exist_ok=True)

img = cv2.imread('assets/food.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_avg = np.mean(img_rgb, axis=2).astype(np.uint8)
gray_weighted = (0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]).astype(np.uint8)
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('outputs/gray_average.jpg', gray_avg)
cv2.imwrite('outputs/gray_weighted.jpg', gray_weighted)
cv2.imwrite('outputs/gray_cv.jpg', gray_cv)

plt.figure(figsize=(12,6))
plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')
plt.subplot(1,4,2)
plt.title("Gray - Avg")
plt.imshow(gray_avg, cmap='gray')
plt.axis('off')
plt.subplot(1,4,3)
plt.title("Gray - Weighted")
plt.imshow(gray_weighted, cmap='gray')
plt.axis('off')
plt.subplot(1,4,4)
plt.title("Gray - cv2")
plt.imshow(gray_cv, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/gray_comparison.png")
plt.show()

# ----------------------------------------------------------------------
# Subtask 2: Grayscale to Black & White (Binary Image)
# ----------------------------------------------------------------------

# Manual Thresholding
threshold_value = 128
bw_manual = (gray_cv > threshold_value).astype(np.uint8) * 255

# Built-in using OpenCV
_, bw_cv = cv2.threshold(gray_cv, threshold_value, 255, cv2.THRESH_BINARY)

# Alternative: Adaptive Threshold
bw_adaptive = cv2.adaptiveThreshold(
    gray_cv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3
)

_, bw_otsu = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('outputs/bw_otsu.jpg', bw_otsu)


# Save all binary images
cv2.imwrite('outputs/bw_manual.jpg', bw_manual)
cv2.imwrite('outputs/bw_cv.jpg', bw_cv)
cv2.imwrite('outputs/bw_adaptive.jpg', bw_adaptive)

# Display all
plt.figure(figsize=(15,6))

plt.subplot(1,5,1)
plt.title("Grayscale")
plt.imshow(gray_cv, cmap='gray')
plt.axis('off')

plt.subplot(1,5,2)
plt.title("Manual (128)")
plt.imshow(bw_manual, cmap='gray')
plt.axis('off')

plt.subplot(1,5,3)
plt.title("cv2.threshold (128)")
plt.imshow(bw_cv, cmap='gray')
plt.axis('off')

plt.subplot(1,5,4)
plt.title("Adaptive (Gaussian)")
plt.imshow(bw_adaptive, cmap='gray')
plt.axis('off')

plt.subplot(1,5,5)
plt.title("Otsu")
plt.imshow(bw_otsu, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('outputs/bw_all_comparison.png')
plt.show()

# ----------------------------------------------------------------------
# Subtask 3: Show One Plane at a Time (Others Set to Zero)
# ----------------------------------------------------------------------

r_plane_only = np.zeros_like(img_rgb)
r_plane_only[:, :, 0] = img_rgb[:, :, 0]

g_plane_only = np.zeros_like(img_rgb)
g_plane_only[:, :, 1] = img_rgb[:, :, 1]

b_plane_only = np.zeros_like(img_rgb)
b_plane_only[:, :, 2] = img_rgb[:, :, 2]

# Save each plane image
cv2.imwrite('outputs/red_only.jpg', cv2.cvtColor(r_plane_only, cv2.COLOR_RGB2BGR))
cv2.imwrite('outputs/green_only.jpg', cv2.cvtColor(g_plane_only, cv2.COLOR_RGB2BGR))
cv2.imwrite('outputs/blue_only.jpg', cv2.cvtColor(b_plane_only, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Red Plane")
plt.imshow(r_plane_only)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Green Plane")
plt.imshow(g_plane_only)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Blue Plane")
plt.imshow(b_plane_only)
plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/planes_comparison.png")
plt.show()
