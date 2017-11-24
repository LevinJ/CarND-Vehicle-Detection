import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip
from heatmap import get_heat_map


class DetectionInVideo():
    
    def __init__(self):
        self.count = 0
        self.debug_frame = False
        self.debug_frame_id = 1178
       
        return
    
    def process_image(self, initial_img):
        try:
            if self.debug_frame and (self.debug_frame_id != self.count):
                self.count = self.count + 1
                return initial_img
            
            print('frame {}'.format(self.count))
             
            final_img= get_heat_map(initial_img)
           
            cv2.putText(final_img,"Frame " + str(self.count),(100,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

            
            self.count = self.count + 1
#             plt.imshow(final_img)
        except:
            plt.imsave('exception_img.jpg', final_img)
            raise 
        return  final_img

    
    def test_on_one_image(self, img_file_path):
        #load the image
        initial_img = cv2.imread(img_file_path)
#         plt.imshow(initial_img)
        final_img = self.process_image_RGB(initial_img)

        plt.imshow(final_img )

        return
    def test_on_images(self):
        img_file_paths = ['solidWhiteCurve.jpg',
                             'solidWhiteRight.jpg',
                             'solidYellowCurve.jpg',
                             'solidYellowCurve2.jpg',
                             'solidYellowLeft.jpg',
                             'whiteCarLaneSwitch.jpg']
        img_file_paths = ['../test_images/'+ file_path for file_path in img_file_paths]
        for img_file_path in img_file_paths:
            print("process image file {}".format(img_file_path))
            self.process_image_file_path(img_file_path)
        print("Done with processing images")
#             break
            
        
        return
    def test_on_videos(self, input_video, output_video):
        clip1 = VideoFileClip(input_video)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(output_video, audio=False)
        return
    def test_on_frame(self):
        clip = VideoFileClip('../data/project_video.mp4')
        initial_img = None
        frame_ids = [740,719]
        for img in clip.iter_frames():
            if self.count in frame_ids:
                initial_img = img
                plt.imsave('../data/falsepositives/frame_{}.jpg'.format(self.count), initial_img)
            self.count = self.count + 1
            
          
        final_img = self.process_image_RGB(initial_img)
        plt.imshow(final_img)
        
        return final_img
    def run(self):
#         self.test_on_videos('./data/test_video_1.mp4','./data/test_1.mp4')
        self.test_on_videos('./data/project_video.mp4','./data/project.mp4')
#         self.test_on_frame()
#         plt.show()
        
        return






if __name__ == "__main__":   
    obj= DetectionInVideo()
    obj.run()