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

    # cv2.imwrite("image_with_blurred_edges.jpg", final_image)
    # cv2.imshow("Blurred Edges", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_image


def test(img):
    # print(img)
    # image_path = img
    image = img.copy()

    scale = 0.5
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        largest_contour = contours[0]

        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped_bottle = image[y:y+h, x:x+w]

        output_image = image.copy()
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(output_image, [largest_contour], -1, (0, 0, 255), 2)

        # cv2.imwrite("bottle_detected.jpg", output_image)
        # cv2.imwrite("bottle_cropped.jpg", cropped_bottle)

        cv2.imshow("Detected Bottle", output_image)
        cv2.imshow("Cropped Bottle", cropped_bottle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No bottle detected in the image.")

    cv2.imshow("image", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_bottle


image_path = "7cc1a5e8-9eb9-4bd9-93c3-676ba3da9eb8.jpg"
image = edges(image_path)
test(image)