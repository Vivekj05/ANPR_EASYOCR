from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr
import os

def home(request):
    #text=request.GET.get('text',None)
    return render(request,'index.html')

def plate(request):
    img_path = request.GET.get('text', None)

    if not img_path:
        return HttpResponse("Error: No file path provided. Please enter a valid image path.")

    if not os.path.exists(img_path):
        return HttpResponse(f"Error: File not found at path '{img_path}'. Please check the path.")

    # ANPR Code
    img = cv2.imread(img_path)
    if img is None:
        return HttpResponse(f"Error: Unable to read the image file at '{img_path}'. Please check the file format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 50, 400)
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints) 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    location = None
    for contour in contours:
    # cv2.approxPolyDP returns a resampled contour, so this will still return a set of (x, y) points
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask = mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y)) 
    (x2, y2) = (np.max(x), np.max(y))
    # Adding Buffer
    cropped_image = gray[x1:x2+3, y1:y2+3]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text_result = " ".join([text for (_, text, _) in result])
    return render(request, 'plate.html', {'result': text_result})
