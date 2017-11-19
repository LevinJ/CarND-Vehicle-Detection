import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from dumpload import DumpLoad
from hog_classifier import extract_features
from sliding_window import draw_boxes
from stackimage import StackImage
from mstimer import MSTimer
import glob





# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    
    draw_img = np.copy(img)
    
    img_tosearch = img[ystart:ystop,:,:]
    
    if scale != 1:
        imshape = img_tosearch.shape
        ctrans_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    
    car_features = extract_features([ctrans_tosearch],feature_vec=False)[0]
    hog1 = car_features[0]
    hog2 = car_features[1]
    hog3 = car_features[2]
    
    on_windows = []
    on_scores = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell


            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features.reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            score = svc.decision_function(test_features)[0]
            #7) If positive (prediction == 1) then save the window
            if score > 0:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bdbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                on_windows.append(bdbox)
                on_scores.append(score)
                
    return on_windows,on_scores


if __name__ == "__main__": 

    model_path = './data/smvmodel.pickle'
    dump_load = DumpLoad(model_path)
    svc, X_scaler = dump_load.load()    
    ystart = 400
    ystop = 656
    scale = 1.5
    orient=9 
    pix_per_cell=8
    cell_per_block=2
    
#     img = mpimg.imread('./data/test_images/test1.jpg')
#         
#     hot_windows,hot_scores = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block)
#     out_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6,scores=hot_scores)   
#     
#     plt.imshow(out_img)
#     plt.show()
    images = glob.glob('./data/hard_frames/*.jpg', recursive=True)
    images = np.random.choice(images, 5)
    
    window_imgs = []
    for image in images:
        tr = MSTimer()
        
        image = mpimg.imread(image)
        draw_image = np.copy(image)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        
        y_start_stop = [380, None] # Min and max in y to search in slide_window()

        
        
        
        hot_windows,hot_scores = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block)                       
        
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6,scores=hot_scores)   
        window_imgs.append(window_img)
        print("duration: {:.1f} seconds".format(tr.stop_timer()))
                         
    
    si = StackImage()
    img_show = si.stack_image_vertical(window_imgs)
    plt.imshow(img_show)
    plt.show()
    
    
    