import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# PART 1: Bit Plane Extraction
# -------------------------------------------------------------------
os.makedirs("outputs/bit_planes", exist_ok=True)

# Load grayscale image
img = cv2.imread('assets/food.jpg', cv2.IMREAD_GRAYSCALE)

bit_planes = []
for i in range(8):
    plane = (img >> i) & 1
    plane_img = (plane * 255).astype(np.uint8)
    bit_planes.append(plane_img)
    cv2.imwrite(f"outputs/bit_planes/bit_plane_{i}.png", plane_img)

# Display all bit planes
plt.figure(figsize=(12, 6))
for i, plane in enumerate(bit_planes):
    plt.subplot(2, 4, i+1)
    plt.imshow(plane, cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/bit_planes/bit_planes_grid.png")
plt.show()

# -------------------------------------------------------------------
# PART 2: Original vs LSB Removed
# -------------------------------------------------------------------
os.makedirs("outputs/bit_slicing", exist_ok=True)

# Remove LSB (bit 0)
lsb_removed = img & 254  # mask 11111110 in binary

cv2.imwrite("outputs/bit_slicing/original_gray.png", img)
cv2.imwrite("outputs/bit_slicing/lsb_removed.png", lsb_removed)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lsb_removed, cmap='gray')
plt.title("LSB Removed")
plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/bit_slicing/og_vs_lsb_removed.png")
plt.show()

# -------------------------------------------------------------------
# PART 3: Invisible Watermarking ("A")
# -------------------------------------------------------------------
# Create watermark pattern (letter A)
watermark = np.zeros_like(img, dtype=np.uint8)
cv2.putText(
    watermark, "A", (img.shape[1]//3, img.shape[0]//2),
    cv2.FONT_HERSHEY_SIMPLEX, 8, 255, thickness=8
)

# Embed watermark in LSB plane
watermarked_img = (img & 254) | ((watermark > 0).astype(np.uint8))

cv2.imwrite("outputs/bit_slicing/watermarked_img.png", watermarked_img)
cv2.imwrite("outputs/bit_slicing/watermark_pattern.png", watermark)

# Show watermark pattern visibly
plt.imshow(watermark, cmap='gray')
plt.title("Watermark Pattern (Letter A)")
plt.axis('off')
plt.savefig("outputs/bit_slicing/watermark_pattern_visible.png")
plt.show()

# -------------------------------------------------------------------
# PART 4: Watermark Extraction
# -------------------------------------------------------------------
extracted_watermark = (watermarked_img & 1) * 255
cv2.imwrite("outputs/bit_slicing/extracted_watermark.png", extracted_watermark)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(watermarked_img, cmap='gray')
plt.title("Watermarked Image (Invisible)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(extracted_watermark, cmap='gray')
plt.title("Extracted Watermark")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(watermark, cmap='gray')
plt.title("Original Watermark Pattern")
plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/bit_slicing/watermark_embedding_extraction.png")
plt.show()
