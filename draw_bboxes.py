import numpy as np
import cv2
import matplotlib.pyplot as plt



# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for bbox in bboxes:
        pt1,pt2 = bbox
        cv2.rectangle(draw_img, pt1, pt2, color=color, thickness=thick)
    return draw_img # Change this line to return image copy with boxes

if __name__ == "__main__": 
    image = cv2.imread('./data/sample/bbox-example-image.jpg')
    # Add bounding boxes in this format, these are just example coordinates.
    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 600))]
    
    result = draw_boxes(image, bboxes)
    plt.imshow(result[...,::-1])
    plt.show()