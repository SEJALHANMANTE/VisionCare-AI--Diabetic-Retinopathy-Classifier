import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("Dataset/0/10_left.jpeg")

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray)

# Save the images
cv2.imwrite("10_left_gray.jpeg", img_gray)  # Save original grayscale image
cv2.imwrite("10_left_clahe.jpeg", img_clahe)  # Save CLAHE-enhanced image

# Plot both images side by side at the same scale
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original Grayscale Image
ax[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Original Grayscale Image")
ax[0].axis('off')

# CLAHE Enhanced Image
ax[1].imshow(img_clahe, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("CLAHE Enhanced Image")
ax[1].axis('off')

plt.tight_layout()
plt.show()
