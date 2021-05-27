# standard imports
import cv2
import tensorflow as tf
import numpy as np
import colorsys
import random

""" 
Utility functions to for training, pre and post processing
"""

def read_class_names(class_file_name):
        
    """ function to load class names from a file """
    
    # open class text file
    with open(class_file_name, 'r') as f:
        
        # intialise empty list to store names
        names = []
        
        # iterate over class names
        for name in f:
            
            # append class name from each line
            names.append(name.strip('\n'))

    return names

def transform_images(image, size):
        
    """ transform image to specified input shape for training and scale pixel values to be between 0 and 1 """

    # convert image to tensor
    image = tf.convert_to_tensor(image)
    
    # add batch dimension to image
    image = tf.expand_dims(image, axis = 0)
    
    # resize image to specified size
    image = tf.image.resize_with_pad(image, size, size)
    
    # standardize image
    image = image / 255
    
    # remove batch dimension
    image = tf.squeeze(image, axis = 0)
    
    return image.numpy()

def decode(yolo_output, num_of_anchor_bbox, classes, strides, anchors, index):

    """ function to decode the outputs from yolo_v3 """
    """ takes in tensor of shape (batch_size, gridsize_x, gridsize_y, number of anchor boxes, number of classes) """
    """ returns tesnor of shape (batch_size, gridsize_x, gridsize_y, number of anchor boxes, number of classes) """
    
    # takes in original anchors and process to scaled anchors based on strides for respective scales
    anchors_scaled = (np.array(anchors).T/strides).T
    
    # obtain dimensions from yolo_output
    conv_shape = tf.shape(yolo_output)
    batch_size = conv_shape[0]
    grid_size = conv_shape[1:3]

    # reshape yolo_output
    yolo_output = tf.reshape(yolo_output, (batch_size, grid_size[0], grid_size[1], num_of_anchor_bbox, 5 + classes))

    # split yolo_output along last axis to extract features
    raw_dx_dy, raw_dw_dh, raw_objectiveness, raw_class_probs = tf.split(yolo_output, (2, 2, 1, classes), axis = -1)

    # create grid where grid[x][y] == (y, x)
    xy_grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))

    # reshape to [gx, gy, 1, 2] and cast to float32 data type
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis = -1), axis = 2) 
    xy_grid = tf.cast(xy_grid, tf.float32)

    # calculate the center position of the prediction box (train_input_size):
    pred_xy = (tf.sigmoid(raw_dx_dy) + xy_grid) * strides[index]

    # calculate the length and width of the prediction box (train_input_size):
    pred_wh = (tf.exp(raw_dw_dh) * anchors_scaled[index]) * strides[index]

    # concatenate pred_xy and pred_wh
    pred_xywh = tf.concat([pred_xy, pred_wh], axis = -1)

    # objectiveness score
    pred_objectiveness = tf.sigmoid(raw_objectiveness) 

    # class probabilities
    pred_prob = tf.sigmoid(raw_class_probs) 

    # concatenate decoded results
    pred = tf.concat([pred_xywh, pred_objectiveness, pred_prob], axis = -1)

    return pred

def bbox_iou(boxes1, boxes2):

    """ function to determine iou from 2 boxes for tensors """

    # obtain area of from the 2 boxes
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # obtain boxes where properties are (x_min, y_min, x_max, y_max)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis = -1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis = -1)

    # obtain maximum coordinates amongst 2 box at top left corner
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])

    # obtain minimum coordinates amongst 2 box at bottom right corner
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # obtain a positive intersection 
    inter_section = tf.maximum(right_down - left_up, 0.0)

    # obtain intersection area 
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # obtain union area 
    union_area = boxes1_area + boxes2_area - inter_area

    # return iou
    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):

    """ function to determine giou (generalised iou) from 2 boxes """

    # obtain boxes where properties are (x_min, y_min, x_max, y_max)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis = -1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis = -1)

    # obtain boxes where properties are (x_min, y_min, x_max, y_max)
    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis = -1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis = -1)

    # obtain area of from the 2 boxes
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # obtain maximum coordinates amongst 2 box at top left corner
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])

    # obtain minimum coordinates amongst 2 box at bottom right corner
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # obtain a positive intersection 
    inter_section = tf.maximum(right_down - left_up, 0.0)

    # obtain intersection area 
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # obtain union area 
    union_area = boxes1_area + boxes2_area - inter_area

    # calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex 
    # surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def loss_func(pred, conv, label, bboxes, num_classes, train_input_size, iou_loss_threshold):

    """ loss function to compiute losses comprising of giou, objectiviness and class probs losses for training """
    """ giou replaces l2 norm losses of x, y, w, h as an improvement from original yolo_v3 """
    
    # obtain number of classes
    num_classes = num_classes
    
    # obtain shape of raw yolo_v3 output (pre-decode)
    conv_shape  = tf.shape(conv)
    
    # obtain batch size of raw yolo_v3 output (pre-decode)
    batch_size  = conv_shape[0]
    
    # obtain output size of raw yolo_v3 output (pre-decode)
    output_size = conv_shape[1]
    
    # obtain train input size
    train_input_size = tf.cast(train_input_size, tf.float32)
    
    # reshape raw conv output 
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))
    
    # obtain objectiveness scores and class probabilites for batch from raw conv output
    conv_raw_objectiveness = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    
    # obtain predicted x, y, w, h and objectiveness scores for batch based on train_input_size post decode
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    
    # obtain label x, y, w, h and objectiveness scores for batch based on train_input_size
    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]
    
    # obtain giou between predictions and labels 
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis = -1)

    # loss factor that gives higher weight to smaller boxes 
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (train_input_size ** 2)
    
    # obtain giou loss 
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    
    # obtain iou between predictions and labels 
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    
    # find the value of iou with the largest prediction box
    max_iou = tf.reduce_max(iou, axis = -1, keepdims = True)

    # if the largest iou is less than the threshold, it is considered that the prediction box contains no objects, 
    # then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_threshold, tf.float32)
    
    # focal factor on objectiveness loss 
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # calculate the objectiveness loss 
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 
    # when there is no object.
    conf_loss = conf_focal * (respond_bbox + respond_bgd) * \
                tf.nn.sigmoid_cross_entropy_with_logits(labels = respond_bbox, logits = conv_raw_objectiveness)
         
    # class probabilities loss
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels = label_prob, logits = conv_raw_prob)
    
    # sum up losses and take mean accross batch
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis = [1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis = [1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis = [1,2,3,4]))
    
    if np.isnan(giou_loss):
        
        giou_loss = tf.Variable(0, trainable = False, dtype = tf.float32)
    
    return giou_loss, conf_loss, prob_loss

def postprocess_boxes(pred_bbox, original_image, train_input_size, score_threshold):
    
    """ function to scale bboxes from train input size to original image size and remove bboxes with low scores """
    
    # valid scle for box
    valid_scale=[0, np.inf]
    
    # turn bbox to array
    pred_bbox = np.array(pred_bbox)
    
    # obtain predicted x, y, w, h, objectiveness score, class probabilities
    pred_xywh = pred_bbox[:, 0:4]
    pred_objectiveness = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    
    # 1. (x, y, w, h) --> (x_org, y_org, w_org, h_org)
    # obtain original image width and height
    org_h, org_w = original_image.shape[:2]
    
    # obtain resize ratio for height and width 
    resize_ratio_h = train_input_size / org_h
    resize_ratio_w = train_input_size / org_w
    
    # scale x, y, w, h to original x, y, w, h
    pred_coor = np.concatenate([np.expand_dims(pred_xywh[:, 0] / resize_ratio_w, axis = -1), 
                                np.expand_dims(pred_xywh[:, 1] / resize_ratio_h, axis = -1),
                                np.expand_dims(pred_xywh[:, 2] / resize_ratio_w, axis = -1),
                                np.expand_dims(pred_xywh[:, 3] / resize_ratio_h, axis = -1),], axis = -1)
  
    # 2. (x_org, y_org, w_org, h_org) --> (xmin_org, ymin_org, xmax_org, ymax_org)
    # obtain diagonal image coordinates
    pred_coor = np.concatenate([pred_coor[:, :2] - pred_coor[:, 2:] * 0.5,
                                pred_coor[:, :2] + pred_coor[:, 2:] * 0.5], axis = -1)

    # 3. clip some boxes those are out of range
    # clip bboxes where xmin_org, ymin_org < 0 and xmax_org, ymax_org out of bounds
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis = -1)
    
    # mask that ensure that if xmin < xmax, ymin /> ymax and vice versa
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis = -1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    # obtain index of class with max prob for each bbox
    classes = np.argmax(pred_prob, axis = -1)
    
    # multiply max prob with objectivness score for each bbox
    scores = pred_objectiveness * pred_prob[np.arange(len(pred_coor)), classes]
    
    # obtain score mask based on score threshold
    score_mask = scores > score_threshold
    
    # obtain combined mask
    mask = np.logical_and(scale_mask, score_mask)
    
    # obtain coordinates, scores and classes after mask
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    
    # return concatenated results 
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis = -1)

def nms(bboxes, iou_threshold, sigma = 0.3, method = 'nms'):
    
    """ function to implement non-maximal suppression / softmax non-maximal supression of bboxes """
    """ takes bboxes with the shape of (num_of_box, 6), where 6 => (xmin, ymin, xmax, ymax, score, class) """
    
    # remove duplicates in classes
    classes_in_img = list(set(bboxes[:, 5]))
    
    # initialise list to store best bboxes
    best_bboxes = []
    
    # iterate over each class
    for cls in classes_in_img:
        
        # get mask for bboxes with the same class and apply on bboxes to obtain array of bboxes with same class
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
       
        # iterate while there are still bboxes in cls_bboxes
        while len(cls_bboxes) > 0:
            
            # select index of the bbox with the highest score 
            max_ind = np.argmax(cls_bboxes[:, 4])
            
            # select bbox with highest score 
            best_bbox = cls_bboxes[max_ind]
            
            # append to best _bbox list 
            best_bboxes.append(best_bbox)
            
            # obtain cls_bboxes without best bbox
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            
            # calculate iou of remaining bboxes with best bbox 
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            
            weight = np.ones((len(iou), ), dtype = np.float32)
            
            # assert method to be either 'nms' or 'soft_nms'
            assert method in ['nms', 'soft_nms']
            
            if method == 'nms':
                
                # obtain nms iou mask based on threshold
                iou_mask = iou > iou_threshold
                
                # apply mask on weights
                weight[iou_mask.numpy()] = 0.0
                
            if method == 'soft_nms':
                
                # obtain soft_nms weights
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            
            # apply weights on cls_bboxes
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            
            # obtain score mask of scores greater than zero
            score_mask = cls_bboxes[:, 4] > 0.
            
            # apply mask on cls_bboxes 
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def draw_bbox(image, bboxes, classes_file_path, show_label = True, show_confidence = True, Text_colors = (255,255,0), 
              rectangle_colors = '', tracking = False):
    
    """ function to draw bboxes on image """
    
    # obtain list of classes name 
    classes = read_class_names(classes_file_path)
    
    # obtain length of classes 
    num_classes = len(classes)
    
    # obtain shape of image
    image_h, image_w, _ = image.shape
    
    # obtain list of unique hsv (hue, saturation, value) for each class
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    
    # obtain unique rgb tuples from hsv tuples
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    
    # scale rgb from 0-1 to 0-255 
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    # shuffle colors list with same seed
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    
    # iterate over bbox in bboxes
    for i, bbox in enumerate(bboxes):
        
        # obtain coordinates of bbox
        coor = np.array(bbox[:4], dtype = np.int32)
        
        # obtain objectiveness score
        score = bbox[4]
        
        # obtain class index
        class_ind = int(bbox[5])
        
        # choose rectangle color if none is given, else chose from tuple
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        
        # obtain thickness of bboxes
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        
        # obtain font scale
        fontScale = 0.75 * bbox_thick
        
        # obtain tuples of min and max coordinates
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # generate bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)
        
        # if show label is true
        if show_label:
            
            # get objectiveness score label
            score_str = " {:.2f}".format(score) if show_confidence else ""
            
            # if tracking show whole score without rounding
            if tracking: score_str = " " + str(score)
                
            # obtain label of class name with objectiveness score
            label = "{}".format(classes[class_ind]) + score_str
                
            # get text size 
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness = bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, 
                          thickness = cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType = cv2.LINE_AA)

    return image

def detect_image(yolo_v3_model, image_paths, batch_frames, output_path, train_input_size, classes_file_path, 
                 score_threshold, iou_threshold, num_of_anchor_bbox, strides, anchors, show = False, 
                 rectangle_colors = ''):
    
    """ function to take in image and apply bbox on it """
    
    # obtain number of classes
    num_of_classes = len(read_class_names(classes_file_path))
    
    # create list to store images
    original_images = []
    
    # iterate over images in chronological order (last image is image of interest to put bbox)
    for x in range(batch_frames):
    
        # obtain original image
        original_image = cv2.imread(image_paths[x])
       
        # append original image to original_images list
        original_images.append(original_image[:])
        
        # convert original image to grayscale 
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
       
        # preprocess image
        image = transform_images(image[:], train_input_size)

        # obtain concat frame if none exist
        if x == 0: 

            concat_image = image[:]

        # concatenate subsequent frames to concat_image
        else:

            concat_image = np.concatenate((concat_image, image), axis = -1)
        
    # add batch dimensions to concatenated image 
    concat_image = concat_image[np.newaxis, ...].astype(np.float32)
    
    # create constant tensor from concatenated image and feed it to yolo_v3_model
    batched_input = tf.constant(concat_image)
    yolo_output = yolo_v3_model(batched_input)
    
    # list to store bboxes from respective scales
    pred_bbox = []
    
    # iterate over 3 scales
    for i in range(3):

        # decode resepctive yolo_output from each scale
        pred_result = decode(yolo_output = yolo_output[i], num_of_anchor_bbox = num_of_anchor_bbox, 
                             classes = num_of_classes, strides = strides, anchors = anchors, index = i)
      
        # obtain results of shape (:, 5 + num_classes), i.e all bboxes
        pred_result_reshaped = tf.reshape(pred_result, (-1, tf.shape(pred_result)[-1]))
    
        # append to pred_bbox
        pred_bbox.append(pred_result_reshaped)
    
    # concatenate all bboxes from all scales
    pred_bbox = tf.concat(pred_bbox, axis = 0)
    
    # post process all bboxes using latest image in orignal_images
    bboxes = postprocess_boxes(pred_bbox, original_images[-1], train_input_size, score_threshold)
   
    # non maximal supression for bboxes
    bboxes = nms(bboxes, iou_threshold, method = 'nms')
    
    # draw bbox on latest image in orignal_images
    image = draw_bbox(original_images[-1], bboxes, classes_file_path, rectangle_colors = rectangle_colors)
    
    # save image if path to save is given
    if output_path != '': cv2.imwrite(output_path, image)
    
    # display image if show is true 
    if show:
        
        # show the image
        cv2.imshow("predicted image", image)
        
        # load and hold the image
        cv2.waitKey(0)
        
        # to close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image

def detect_video(yolo_v3_model, video_path, batch_frames, output_path, train_input_size, classes_file_path, 
                 score_threshold, iou_threshold, num_of_anchor_bbox, strides, anchors, show = False, 
                 rectangle_colors = ''):
    
    """ function to take in video and apply bbox on it """
    
    # obtain number of classes
    num_of_classes = len(read_class_names(classes_file_path))
    
    # obtain VideoCapture object 
    vid = cv2.VideoCapture(video_path)
    
    # obtain width, height and fps of video
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # obtain video codec
    codec = cv2.VideoWriter_fourcc(*'XVID')
    
    # obtain output_path
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps+1, (width, height)) 

    # create list to store images
    images = []
    
    # variable to track frame
    frame = 0 
    
    while True:
        
        try:
        
            # grabs, decodes and returns the next video frame
            _, image = vid.read()
            
            # append original image to original_images list
            images.append(image[:])
            
            # increment frame
            frame += 1
            
             
            # if current frame is less than batch_frames
            if frame < batch_frames:
                
                # move to next frame 
                continue
            
            # iterate over images in chronological order (last image is image of interest to put bbox)
            for x in range(batch_frames):
                
                # convert original image to grayscale 
                image = cv2.cvtColor(images[-batch_frames + x + 1], cv2.COLOR_BGR2RGB)
    
                # preprocess image
                image = transform_images(image[:], train_input_size)
    
                # obtain concat frame if none exist
                if x == 0: 
    
                    concat_image = image[:]
    
                # concatenate subsequent frames to concat_image
                else:
    
                    concat_image = np.concatenate((concat_image, image), axis = -1)
        
        except:
            
            break
        
        # add batch dimensions to concatenated image 
        concat_image = concat_image[np.newaxis, ...].astype(np.float32)
        
        # create constant tensor from concatenated image and feed it to yolo_v3_model
        batched_input = tf.constant(concat_image)
        yolo_output = yolo_v3_model(batched_input)

        # list to store bboxes from respective scales
        pred_bbox = []

        # iterate over 3 scales
        for i in range(3):

            # decode resepctive yolo_output from each scale
            pred_result = decode(yolo_output = yolo_output[i], num_of_anchor_bbox = num_of_anchor_bbox, 
                                 classes = num_of_classes, strides = strides, anchors = anchors, index = i)

            # append to pred_bbox
            pred_bbox.append(pred_result)
        
        # obtain results of shape (:, 5 + num_classes), i.e all bboxes
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        
        # concatenate all bboxes from all scales
        pred_bbox = tf.concat(pred_bbox, axis = 0)

        # post process all bboxes using latest image in orignal_images
        bboxes = postprocess_boxes(pred_bbox, images[-1], train_input_size, score_threshold)

        # non maximal supression for bboxes
        bboxes = nms(bboxes, iou_threshold, method = 'nms')

        # draw bbox on latest image in orignal_images
        image = draw_bbox(images[-1], bboxes, classes_file_path, rectangle_colors = rectangle_colors)
        
        # save image frame to video path if path to save is given
        if output_path != '': out.write(image)
        
        # display image frame (i.e play video) if show is true 
        if show:
            
            # show the image
            cv2.imshow('output', image)
            
            # if q key is presssed
            if cv2.waitKey(25) & 0xFF == ord("q"):
                
                # end session
                cv2.destroyAllWindows()
                
                # break out of while loop
                break
    
    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()