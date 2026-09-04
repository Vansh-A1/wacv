import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/home/projectwork/student_package/dreal4_6.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_clean = cv2.GaussianBlur(image, (5, 5), 0)

sobelx = cv2.Sobel(image_clean, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image_clean, cv2.CV_64F, 0, 1, ksize=5)

edges = cv2.Canny(image_clean, 100, 200)

plt.figure(figsize=(24, 18))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image (Grayscale)", fontsize=20)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sobelx, cmap='gray')
plt.title("Sobel X", fontsize=20)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sobely, cmap='gray')
plt.title("Sobel Y", fontsize=20)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(edges, cmap='gray')
plt.title("Edges Detected", fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.savefig("/home/projectwork/student_package/output_plots.png", dpi=150)
plt.show()

print(sobelx)
print(sobely)
