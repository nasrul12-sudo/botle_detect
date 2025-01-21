import cv2
import numpy as np
import tensorflow as tf 
import detec
import cv2

from keras.models import load_model

def load(img, img_height=150, img_width=150):
    model = load_model("botle.h5")
    # print(model)

    resize = cv2.resize(img, (img_height, img_width))
    print(resize)
    normalize = resize / 255.0

    preprocessed = np.expand_dims(normalize, axis=0)

    pred = model.predict(preprocessed)
    print(pred)

def open_img(cropped_bottle, output_image):
        cv2.imwrite("bottle_detected.jpg", output_image)
        cv2.imshow("Detected Bottle", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

image_path = "7cc1a5e8-9eb9-4bd9-93c3-676ba3da9eb8.jpg"

data = detec
image = data.edges(image_path)
images = data.test(image)

img_crp = images[0]
img_org = images[1]

load(img_crp)
open_img(img_crp, img_org)
