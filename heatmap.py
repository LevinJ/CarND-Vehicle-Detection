import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from hog_subsample import find_cars_with_scales
from sliding_window import draw_boxes
from stackimage import StackImage
from mstimer import MSTimer
import glob


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def get_heat_map(image):
    tr = MSTimer()
    
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    box_list, hot_scores = find_cars_with_scales(image)
    bdboxes_img = draw_boxes(np.copy(image), box_list, color=(0, 0, 255), thick=6,scores=hot_scores)
    
    
    heat_1 = np.zeros_like(image[:,:,0]).astype(np.float)  
   
    heat_1 = add_heat(heat_1,box_list)
    print("max bd = {}".format(np.max(heat_1)))
    
    
    heat_1 = np.clip(heat_1, 0, 255).astype(np.uint8)
    
    heat_2 = apply_threshold(np.copy(heat_1), 1)
    heatmap = np.clip(heat_2, 0, 255).astype(np.uint8)
    
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    heat_1 = (heat_1.astype(np.float) * 30).clip(0, 255).astype(np.uint8)
    heatmap = (heatmap.astype(np.float) * 30).clip(0, 255).astype(np.uint8)
    
    heat_show = si.stack_image_horizontal([heat_1, heatmap])
    right_show = si.stack_image_vertical([bdboxes_img, heat_show])
    window_img = si.stack_image_horizontal([draw_img,right_show])
   
    print("duration: {:.1f} seconds".format(tr.stop_timer()))
    return window_img
si = StackImage()
if __name__ == "__main__": 
    
    images = glob.glob('./data/hard_frames/*.jpg', recursive=True)
    images = np.random.choice(images, 5)
#     images = ['./data/hard_frames/frame_622.jpg']
    
    
    window_imgs = []
    for image in images:
        tr = MSTimer()
        
        image = mpimg.imread(image)
        
        
        window_img = get_heat_map(image)
        window_imgs.append(window_img)
        
                         
    
    
    img_show = si.stack_image_vertical(window_imgs)
    plt.imshow(img_show)
    plt.show()
