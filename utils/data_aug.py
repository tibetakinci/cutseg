import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def clahe_f(img):
    # Contrast enhancement (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    output = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return output


def hair_removal(img):
    # Gray scaled image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Creating kernel to run blackhat filtering for enhance dark objects in frame
    kernel = cv2.getStructuringElement(1, (10,10))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Binary threshold image with thresholding value
    img_thresh = 10
    _, thresh = cv2.threshold(blackhat, img_thresh, 255, cv2.THRESH_BINARY)

    # Interpolate thresholded area with predefined radius
    output = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

    return output


def remove_dark_corner(img):
    # Grayscale image of original img size
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Contracted image and grayscale image due to eliminate border line of the frame
    height, width, depth = img.shape
    img_contracted = img[1:height-1, 1:width-1, :]
    gray_contracted = cv2.cvtColor(img_contracted, cv2.COLOR_RGB2GRAY)
    
    # Binary threshold grayscale image
    img_thresh = 50
    _, thresh = cv2.threshold(gray_contracted, img_thresh, 255, cv2.THRESH_BINARY)

    # Find contours in the contracted image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(img_contracted.shape)

    # Sort contours according to descending contour area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    (x,y), radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    radius = int(radius) - 2
    '''
    if radius >= (np.sqrt((height-2)**2+(width-2)**2)/2-2):
        radius = radius + 10
    '''

    # Create mask as a circle with given radius and center point
    mask = cv2.circle(np.ones(gray.shape, dtype=np.uint8)*255, center, radius, (0,0,0), -1)

    # Interpolate non-contracted image with mask
    output = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)

    return output
   

def do_aug(img, clahe, hair):
    img = np.array(img).astype(np.uint8)

    #Hair removal
    if hair:
        img = hair_removal(img)

    #CLAHE
    if clahe:
        img = clahe_f(img)

    return Image.fromarray(img, 'RGB')


if __name__ == "__main__":
    imgs = ['0000042', '0000074', '0000171', '0000000', '0000004', '0000045']
    img_dir = "../../dataset/ISIC2016/train/imgs"

    img_root = f"{img_dir}/ISIC_{imgs[5]}.jpg"

    #hair_removal(img_root)
    #remove_dark_corner(img_root)
    clahe_f(img_root)