import cv2
import numpy as np
import matplotlib.pyplot as plt

def edges(image_path, blur_intensity=25):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2

    radius_x, radius_y = int(width * 0.4), int(height * 0.4)
    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
    
    mask_inverted = cv2.bitwise_not(mask)

    blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)

    sharp_region = cv2.bitwise_and(image, image, mask=mask)
    blurred_region = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inverted)
    final_image = cv2.add(sharp_region, blurred_region)

    return final_image

def test(img):
    image = img.copy()

    scale = 0.5
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    images = []
    if contours:
        largest_contour = contours[0]

        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped_bottle = image[y:y+h, x:x+w]
        images.append(cropped_bottle)

        output_image = image.copy()
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(output_image, [largest_contour], -1, (0, 0, 255), 2)

    else:
        print("No bottle detected in the image.")

    return cropped_bottle, output_image
