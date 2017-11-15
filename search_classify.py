import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from hog_classifier import extract_features

from dumpload import DumpLoad
from sliding_window import slide_window
from sliding_window import draw_boxes
from stackimage import StackImage
from mstimer import MSTimer

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img):    
    
    cars = [img]
    car_features = extract_features(cars)
    return car_features[0]

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def get_all_windows():
    all_windows= []
    xy_windows = [120]
    xyo = 0.75
    for xyw in xy_windows:
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                                xy_window=(xyw, xyw), xy_overlap=(xyo, xyo))
        all_windows.extend(windows)
    return all_windows
    
    
if __name__ == "__main__": 

    model_path = './data/smvmodel.pickle'
    dump_load = DumpLoad(model_path)
    svc, X_scaler = dump_load.load()

    # Check the prediction time for a single sample
    
    images = glob.glob('./data/sample/*.jpg', recursive=True)
    images = ['./data/test_images/test1.jpg']
#     images = glob.glob('./data/test_images/test*.jpg', recursive=True)
#     images = np.random.choice(images, 1)
    
    window_imgs = []
    for image in images:
        tr = MSTimer()
        
        image = mpimg.imread(image)
        draw_image = np.copy(image)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        
        y_start_stop = [None, None] # Min and max in y to search in slide_window()

        
        windows = get_all_windows()
        
        hot_windows = search_windows(image, windows, svc, X_scaler)                       
        
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)   
        window_imgs.append(window_img)
        print("duration: {:.1f} seconds".format(tr.stop_timer()))
                         
    
    si = StackImage()
    img_show = si.stack_image_vertical(window_imgs)
    plt.imshow(img_show)
    plt.show()


