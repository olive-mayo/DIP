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
