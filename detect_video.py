from yolo_v3_model import yolo_v3
from utils import read_class_names, detect_video

"""
function to detect video, show and saved video with predictions
"""

CLASSES               = "thermographic_data/classes.txt"
NUM_OF_CLASSES        = len(read_class_names(CLASSES))
MODEL_NAME            = "model_2"
CHECKPOINTS_FOLDER    = "checkpoints" + "/" + MODEL_NAME
ANNOT_PATH            = "raw_videos/free_2.mp4"
OUTPUT_PATH           = 'predicted_videos/' + MODEL_NAME 
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
    
    # obtain name of test from annotation path
    test_file_name = ANNOT_PATH.rsplit('/')[-1]
   
    # obtain output path
    output_path = OUTPUT_PATH + "/" + test_file_name
    
    # detect video
    detect_video(yolo_v3_model = yolo_v3_model, video_path = ANNOT_PATH, batch_frames = YOLO_BATCH_FRAMES, 
                  output_path = output_path, train_input_size = INPUT_SIZE, classes_file_path = CLASSES, 
                  score_threshold = SCORE_THRESHOLD, iou_threshold = IOU_THRESHOLD, 
                  num_of_anchor_bbox = YOLO_ANCHOR_PER_SCALE, strides = YOLO_STRIDES, 
                  anchors = YOLO_ANCHORS, show = False, rectangle_colors = '')

if __name__ == '__main__':
    
    main()