import os
import random
from yolo_v3_model import yolo_v3
from utils import read_class_names, detect_image

"""
function to detect images, show and saved images with predictions
"""

CLASSES               = "thermographic_data/classes.txt"
NUM_OF_CLASSES        = len(read_class_names(CLASSES))
MODEL_NAME            = "model_2"
CHECKPOINTS_FOLDER    = "checkpoints" + "/" + MODEL_NAME
ANNOT_PATH            = "thermographic_data/test/images/pr"
OUTPUT_PATH           = 'predicted_images/' + MODEL_NAME + "/pr"
DETECT_BATCH          = False
DETECT_WHOLE_VID      = True
BATCH_SIZE            = 1804
IMAGE_PATH            = ANNOT_PATH + "/free_3/free_3_frame_100"
INPUT_SIZE            = 416
SCORE_THRESHOLD       = 0.8
IOU_THRESHOLD         = 0.45

# YOLO options
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_BATCH_FRAMES           = 5
YOLO_PREPROCESS_IOU_THRESH  = 0.3
YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

def main():
    
    # create the yolo_v3_model
    yolo_v3_model = yolo_v3(num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, classes = NUM_OF_CLASSES, 
                            checkpoint_dir = CHECKPOINTS_FOLDER, model_name = MODEL_NAME)
    
    # load weights of last saved checkpoint
    yolo_v3_model.load_weights(yolo_v3_model.checkpoint_path).expect_partial()
        
    # use model to detect random batch of images from validate / test
    if DETECT_BATCH:
        
        # variable to track images selected
        batch = 0 
        
        # list to store path to images
        images_paths = []
        
        # obtain name of the tests that are the same for image and labels directories
        name_of_tests = [name for name in os.listdir(ANNOT_PATH) 
                         if os.path.isdir(os.path.join(ANNOT_PATH, name))] 
        
        # iterate while images selected is less that batch size
        while batch != BATCH_SIZE:
        
            # iterate over each test
            for test_name in name_of_tests:
                
                # obtain path for images and labels for specific test
                annot_path_images_test = ANNOT_PATH + "/" + test_name
                
                # select number of images to be selected for specific test
                num_of_images = random.randint(0, BATCH_SIZE - batch)
                
                # obtain the number of frames for specific test (not including class.txt file)
                num_of_image_files = len([name for name in os.listdir(annot_path_images_test) 
                                          if os.path.isfile(os.path.join(annot_path_images_test, name))])
                
                # get list of index of len(num_of_images) of random frames from specific test
                image_frames = random.sample(range(YOLO_BATCH_FRAMES, num_of_image_files), num_of_images)
                
                # iterate over each index
                for frame in image_frames:
                    
                    # intialise image path list
                    image_path = []
                    
                    # iterate over number of batch frames
                    for x in range(YOLO_BATCH_FRAMES):
                        
                        # obtain path of image and label file
                        annot_path_images_test_file = annot_path_images_test + "/" + test_name + "_frame_" + \
                                                      str(frame - YOLO_BATCH_FRAMES + x + 1) + ".jpg"
                        
                        # append selected image to image_path
                        image_path.append(annot_path_images_test_file)
                        
                    # append to image_paths
                    images_paths.append(image_path[:])
                
                # increment batch
                batch += num_of_images
                
        # iterate over selected images
        for x, path in enumerate(images_paths):
            
            if OUTPUT_PATH != '':
            
                # obtain output path
                output_path = OUTPUT_PATH + "/image_" + str(x) + ".jpg"
                
                detect_image(yolo_v3_model = yolo_v3_model, image_paths = path, batch_frames = YOLO_BATCH_FRAMES, 
                             output_path = output_path, train_input_size = INPUT_SIZE, classes_file_path = CLASSES, 
                             score_threshold = SCORE_THRESHOLD, iou_threshold = IOU_THRESHOLD, 
                             num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, strides = YOLO_STRIDES, 
                             anchors = YOLO_ANCHORS, show = False, rectangle_colors = '')
            else:
                
                detect_image(yolo_v3_model = yolo_v3_model, image_paths = path, batch_frames = YOLO_BATCH_FRAMES, 
                             output_path = '', train_input_size = INPUT_SIZE, classes_file_path = CLASSES, 
                             score_threshold = SCORE_THRESHOLD, iou_threshold = IOU_THRESHOLD, 
                             num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, strides = YOLO_STRIDES, 
                             anchors = YOLO_ANCHORS, show = True, rectangle_colors = '')
    
    # detect whole video
    elif DETECT_WHOLE_VID: 
    
        # obtain the number of frames for specific test (not including class.txt file)
        num_of_image_files = len([name for name in os.listdir(ANNOT_PATH) 
                                  if os.path.isfile(os.path.join(ANNOT_PATH, name))])
        
        # obtain name of test from annotation path
        test_name = ANNOT_PATH.rsplit('/')[-1]
        
        # iterate over images beyond batch frames
        for x in range(YOLO_BATCH_FRAMES, num_of_image_files):
            
            # intialise image path list
            image_path = []
            
            # iterate over number of batch frames
            for i in range(YOLO_BATCH_FRAMES):
                
                # obtain path of image and label file
                annot_path_images_test_file = ANNOT_PATH + "/" + test_name + "_frame_" + \
                                              str(x - YOLO_BATCH_FRAMES + i) + ".jpg"
                
                # append selected image to image_path
                image_path.append(annot_path_images_test_file)
               
            # obtain output path
            output_path = OUTPUT_PATH + "/image_" + str(x) + ".jpg"
            
            # detect image
            detect_image(yolo_v3_model = yolo_v3_model, image_paths = image_path, batch_frames = YOLO_BATCH_FRAMES, 
                          output_path = output_path, train_input_size = INPUT_SIZE, classes_file_path = CLASSES, 
                          score_threshold = SCORE_THRESHOLD, iou_threshold = IOU_THRESHOLD, 
                          num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, strides = YOLO_STRIDES, 
                          anchors = YOLO_ANCHORS, show = False, rectangle_colors = '')
            
    
    else:
        
        # detect specific frame
        detect_image(yolo_v3_model = yolo_v3_model, image_paths = IMAGE_PATH, batch_frames = YOLO_BATCH_FRAMES, 
                     output_path = '', train_input_size = INPUT_SIZE, classes_file_path = CLASSES, 
                     score_threshold = SCORE_THRESHOLD, iou_threshold = IOU_THRESHOLD, 
                     num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, strides = YOLO_STRIDES, 
                     anchors = YOLO_ANCHORS, show = True, rectangle_colors = '')
        
if __name__ == '__main__':
    
    main()