import cv2
import tensorflow as tf
import numpy as np
import os
from utils import read_class_names, transform_images, bbox_iou

""" 
Dataset class that process images and annotations to be feed to YOLO_V3 for training / test
"""

class Dataset():
    
    def __init__(self, dataset_type, annot_path, batch_size, train_input_size, strides, classes_file_path, anchors, 
                 anchor_per_scale, max_bbox_per_scale, batch_frames, iou_threshold):
        
        """ class constructor that instantiates the necessary attributes for dataset to feed data during training """
        
        # directory path for labels and images for training or test
        self.annot_path = annot_path
        
        # batch size of images fed to yolo_v3
        self.batch_size = batch_size
        
        # size of input for yolo_v3
        self.train_input_size = train_input_size
        
        # strides used by yolo_v3
        self.strides = np.array(strides)
        
        # list of class names
        self.classes = read_class_names(classes_file_path)
        
        # number of classes
        self.num_classes = len(self.classes)
        
        # takes in original anchors and process to scaled anchors based on strides for respective scales
        self.anchors = (np.array(anchors).T/self.strides).T
        
        # number of anchors per scale
        self.anchor_per_scale = anchor_per_scale
        
        # maximum number of bonding box per scale
        self.max_bbox_per_scale = max_bbox_per_scale
        
        # number of frames in in concatenated image 
        self.batch_frames = batch_frames
        
        # load annotations based on dataset type
        self.annotations = self.load_annotations(dataset_type)
        
        # number of samples
        self.num_samples = len(self.annotations)
        
        # number of batches given batch size
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        
        # counter to track batches processed
        self.batch_count = 0
        
        # threshold for iou
        self.iou_threshold = iou_threshold
        
    def load_annotations(self, dataset_type):
        
        """ function to load the necessary image and label annotations to dataset """
        """ returns list of image paths with past frames and list of annotations as strings """
        
        # initialise list to store all annotations
        final_annotations = []
        
        # annotation path has directories for images and labels
        self.annot_path_images = self.annot_path + "/images"
        self.annot_path_labels = self.annot_path + "/labels"
        
        # obtain name of the tests that are the same for image and labels directories
        name_of_tests = [name for name in os.listdir(self.annot_path_images) 
                         if os.path.isdir(os.path.join(self.annot_path_images, name))] 
        
        # iterate over each test
        for test_name in name_of_tests:
            
            # obtain path for images and labels for specific test
            annot_path_images_test = self.annot_path_images + "/" + test_name
            annot_path_labels_test = self.annot_path_labels + "/" + test_name
            
            # obtain the number of frames for specific test (not including class.txt file)
            num_of_image_files = len([name for name in os.listdir(annot_path_images_test) 
                                      if os.path.isfile(os.path.join(annot_path_images_test, name))])
            
            # initialise list to store image paths in ascending order
            annot_path_images_test_file_list = []
            
            # iterate over each image file in specific test
            for x in range(num_of_image_files):
                
                # obtain path of image and label file
                annot_path_images_test_file = annot_path_images_test + "/" + test_name + "_frame_" + str(x) + ".jpg"
                annot_path_labels_test_file = annot_path_labels_test + "/" + test_name + "_frame_" + str(x) + ".txt"
                
                # append image path to annot_path_images_test_file_list
                annot_path_images_test_file_list.append(annot_path_images_test_file)
                
                # pass iteration if there arent enough image frames from past frames
                if x < self.batch_frames:
                    
                    continue
                
                # try to open label file
                try: 
                    
                    with open(annot_path_labels_test_file, 'r') as f:
                        
                        # read the text file
                        txt = f.readlines()
                        
                        # obtain annotations for each bounding box (per line)
                        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
                
                # ignore file not found error (ignore background images with no objects)
                except FileNotFoundError:
                    
                    pass
                
                # append image paths to specified past frames and annotations to final_annotations
                else:
                
                    final_annotations.append([annot_path_images_test_file_list[-self.batch_frames:], annotations])
        
        return final_annotations
    
    def parse_annotation(self, annotations):
        
        """ function to process annotations to generate image to be fed to model and labels for training """
        
        # obtain array of labels for bounding boxes and classes
        bboxes = np.array([list(map(float, annotation.split())) for annotation in annotations[1]])
        
        # obtain list of image paths of current and past frames
        image_paths = annotations[0]
    
        # iterate over batch of frames (from last to current)
        for x in range(self.batch_frames):
            
            # read image from path
            image = cv2.imread(image_paths[x])
    
            # preprocess image
            image = transform_images(image[:], self.train_input_size)
            
            # obtain concat frame if none exist
            if x == 0: 

                concat_image = image[:]
            
            # concatenate subsequent frames to concat_image
            else:

                concat_image = np.concatenate((concat_image, image), axis = -1)
                
        return concat_image, bboxes
    
    def __iter__(self):
        
        """ function to return the iterator object itself """
        
        return self
    
    def __len__(self):
        
        """ function to return number of batches given batch size """
        
        return self.num_batchs
    
    def __next__(self):
        
        """ function to return next batch of images and labels from iterator """
        
        with tf.device('/cpu:0'):
            
            # output sizes of yolo_v3
            self.train_output_sizes = self.train_input_size // self.strides
            
            # initialise array of zeros based on original image input size to store image
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 
                                    3 * self.batch_frames), dtype = np.float32)
            
            # intialise arrays of zeros based on output grid shape to stores processed labels
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype = np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype = np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype = np.float32)

            # initialise array of zeros to store true x, y, w, h bboxes for respective scales 
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype = np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype = np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype = np.float32)
            
            # count of number of images labels processed
            num = 0
            
            # iterate over number of batches 
            if self.batch_count < self.num_batchs:
                
                # iterate over batch size
                while num < self.batch_size:
                    
                    # obtain index for to select annotations
                    index = self.batch_count * self.batch_size + num
                    
                    # if index exceed number of samples
                    if index >= self.num_samples: 
                        
                        # loop back to beginning of index
                        index -= self.num_samples
                    
                    # obtain annotation
                    annotation = self.annotations[index]
                    
                    # process annotaitons to obtain image and bboxes (labels)
                    image, bboxes = self.parse_annotation(annotation)
                        
                    # obtain labels for 3 outputs and 
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    
                    # append outputs (image and labels) to necessary lists
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    
                    # increase number of images labels processed counter
                    num += 1
                
                # # increase batch counter 
                self.batch_count += 1
                
                # package targets according to yolo_v3 outputs
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            
            else:
                
                # reset batch count
                self.batch_count = 0
                
                # shuffle annotations
                np.random.shuffle(self.annotations)
                
                # stop iteration
                raise StopIteration
        
    def preprocess_true_boxes(self, bboxes):
        
        """ function to process labelled data """
        """ takes in labelled bboxes data of shape """
        
        # initialise list with array of zeros for each scale to store processed labels
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        
        # intialise list with array of zeros for each scale for the specified max number of bboxes per scale 
        # to store processed x, y, w, h of bbox
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        
        # intialise array to count number of bounding boxes for each scale
        bbox_count = np.zeros((3,))
        
        # iterate over labelled bboxes 
        for bbox in bboxes:
            
            # store class index
            bbox_class_ind = int(bbox[0])
          
            # stores true x_norm, y_norm, w_norm, h_norm relative to image height and width
            bbox_coor = bbox[1:]
            
            # one hot encoding of classes
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            
#             # smooth one hot encoding for large number of classes
#             uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
#             theta = 0.01
#             smooth_onehot = onehot * (1 - theta) + theta * uniform_distribution
            
            # obtain true x, y, w, h based on train_input_size
            bbox_xywh = bbox_coor * self.train_input_size
            
            # obtain true scaled x, y, w, h for each stride
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            # empty list to store iou for each scale
            iou = []
            
            # boolean to indicate if there exist anchor bbox that gives sufficient iou over object
            exist_positive = False
            
            # iterate over 3 scales
            for i in range(3):
                
                # initialise array of zeros to store anchors per scale
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                
                # obtain scaled x, y of object
                anchors_xywh[:, 0:2] = bbox_xywh_scaled[i, 0:2]
                
                # get scaled w, h of anchor
                anchors_xywh[:, 2:4] = self.anchors[i]
                
                # obtain iou for between all scaled anchors and scaled true bbox
                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                
                # append iou_scale to iou list
                iou.append(iou_scale)
                
                # check if iou of all anchors is minimally above specified threshold
                iou_mask = iou_scale > self.iou_threshold
                
                # if larger than specified threshold 
                if np.any(iou_mask):
                    
                    # obtain x, y indices for labelled list
                    xind, yind = np.round(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    
                    # store processed labelled data for anchors above specified threshold
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = onehot
                    
                    # obtain index of bbox based on current number of bbbox on specific scale relative to specified 
                    # max number of bboxes per scale
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    
                    # store true x, y, w, h based on original image size
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    
                    # keep count of bboxes from each scale
                    bbox_count[i] += 1
                    
                    # update boolean
                    exist_positive = True
            
            # when there are no anchors across all 3 scales with iou exceeding specified threshold
            if not exist_positive:
                
                # select index of anchor with largest iou
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                
                # obtain index of scale of best anchor
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                
                # obtain index of best anchor 
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                
                # obtain x, y indices for labelled list
                xind, yind = np.round(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                
                # store processed labelled data for anchor with largest iou below specified threshold
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = onehot
                
                # obtain index of bbox based on current number of bbbox on specific scale relative to specified 
                # max number of bboxes per scale
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                
                # store true x, y, w, h based on train_input_size
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                
                # keep count of bboxes from each scale
                bbox_count[best_detect] += 1
        
        # obtain processed labels and bboxes
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes