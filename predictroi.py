import sys
import os
sys.path.insert(0, os.path.abspath('..')) 

from dumpload import DumpLoad
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from hog_classifier import extract_features
import numpy as np

model_path = './data/smvmodel.pickle'
dump_load = DumpLoad(model_path)
clf, scaler = dump_load.load()

def single_img_features(img):    
    
#     colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#     orient = 9
#     pix_per_cell = 8
#     cell_per_block = 2
#     hog_channel =  "ALL"# Can be 0, 1, 2, or "ALL"
    
    cars = [img]
    car_features = extract_features(cars)
    return car_features[0]
class PredictRoi():    

    def __init__(self):
        
        self.refPt = []
        self.cropping = False

        return
   
    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables

     
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True
     
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False
     
            # draw a rectangle around the region of interest
            x1,y1 = self.refPt[0]
            x2,y2 = self.refPt[1]
            
            self.refPt = []
            self.__predict(x1,y1,x2,y2)
        elif event == cv2.EVENT_MOUSEMOVE:
            if not self.cropping:
                return
            x1,y1 = self.refPt[0]
            x2,y2 = x,y
            
            self.__predict(x1,y1,x2,y2)
            
            
        return
    def __predict(self, x1,y1,x2,y2):
        self.image = self.clone.copy()
        width = x2-x1
        height = y2-y1
        print('width: {}, height:{}, ratio {}'.format(width, height, width/float(height)))
        roi = self.clone[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64,64))
        roi = roi[...,::-1]
        cv2.rectangle(self.image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        
        features = single_img_features(roi)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        score = clf.decision_function(test_features)
        
        if score >=0:
            cv2.putText(self.image,str(score),(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            print("car {}:{}, score {}".format(y1,y2, score))
        else:
            print("non car {}:{}, score {}".format(y1,y2,score))
            cv2.putText(self.image,str(score),(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        return

    def predict_roi(self, img_path):
        self.image = cv2.imread(img_path)
        self.clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
        
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF
         
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = self.clone.copy()
         
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        return
    def predict_img(self, img_path):
        img = mpimg.imread(img_path)
        img= self.process_image_RGB(img, None, None,Debug = True)
        #for the purpose of showing hsv value
        self.img = img 
        _, ax = plt.subplots()
        ax.format_coord = self.format_coord
        ax.imshow(img)

        plt.show()
        return
    def format_coord(self, x, y):
        pt = self.img[y, x, :]
        return 'RGB value, x={:.0f}, y={:.0f}  [R={}, G={}, B={}]'.format(x, y, pt[0],pt[1],pt[2])
    
    
    def run(self):
      
       
        img_path = './data/sample/bbox-example-image.jpg'
        img_path = './data/hard_frames/frame_1041.jpg'

        self.predict_roi(img_path)
#         self.predict_img(img_path)
        
        
        


        return
    


if __name__ == "__main__":   
    obj= PredictRoi()
    obj.run()