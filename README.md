# Drone Detection using Thermal Signature
This repository highlights the work for night-time drone detection using a using an Optris PI Lightweight thermal camera. The work is published in the International Conference of Unmanned Air Systems 2021 ([ICUAS 2021](http://www.uasconferences.com/2021_icuas/)) and the paper can be read in detail in [ICUAS_2021_paper](https://https://github.com/mion666459/thermal_signature_drone_detection/blob/main/ICUAS_2021_paper.pdf).

## Requirements

The following are the requirements with Python 3.7.7

	tensorflow==2.4.0
	opencv_contrib_python==4.5.1.48
	numpy==1.20.3	
## Model Architecture

The following diagram highlights the architecture of model based on YOLOV3. However, unlike typical single image object detection, the model takes in the concatenation of a specified number of images in the past relative to the image of interest. This is to encapsulate the motion of the drone as an input feature for detection, a necessity given that thermal signatures of different are generally globular in shape after a certain distance depending on the fidelity of the thermal camera used. Further details can be found in [ICUAS_2021_paper](https://https://github.com/mion666459/thermal_signature_drone_detection/blob/main/ICUAS_2021_paper.pdf).

![Model Architecture](/readme_images/model_architecture.png)



## Training and Testing

Clone the repository, adjust the training/testing parameters in [`train.py`](https://github.com/mion666459/thermal_signature_drone_detection/blob/main/train.py) as shown and execute the code. The training data comprises of data from a controlled indoor environment while the test data contains a mixture data from indoor and outdoor environments. 

```
# Train options
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_CLASSES               = "thermographic_data/classes.txt"
TRAIN_NUM_OF_CLASSES        = len(read_class_names(TRAIN_CLASSES))
TRAIN_MODEL_NAME            = "model_2"
TRAIN_ANNOT_PATH            = "thermographic_data/train" 
TRAIN_LOGDIR                = "log" + '/' + TRAIN_MODEL_NAME
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints" + '/' + TRAIN_MODEL_NAME
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 416
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 10
TRAIN_DECAY                 = 0.8
TRAIN_DECAY_STEPS           = 50.0

# TEST options
TEST_ANNOT_PATH             = "thermographic_data/validate"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45
```

Once the model is trained, you can test the model's predictions on images using [`detect_image.py`](https://github.com/mion666459/thermal_signature_drone_detection/blob/main/detect_image.py). Adjust the the following parameters in [`detect_image.py`](https://github.com/mion666459/thermal_signature_drone_detection/blob/main/detect_image.py) and execute the code.

```
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
```

Similarly, you can test the model's predictions on videos using [`detect_video.py`](https://github.com/mion666459/thermal_signature_drone_detection/blob/main/detect_video.py). Adjust the following parameters in [`detect_video.py`](https://github.com/mion666459/thermal_signature_drone_detection/blob/main/detect_video.py) and execute the code. 

```
CLASSES               = "thermographic_data/classes.txt"
NUM_OF_CLASSES        = len(read_class_names(CLASSES))
MODEL_NAME            = "model_2"
CHECKPOINTS_FOLDER    = "checkpoints" + "/" + MODEL_NAME
ANNOT_PATH            = "raw_videos/free_2.mp4"
OUTPUT_PATH           = 'predicted_videos/' + MODEL_NAME 
INPUT_SIZE            = 416
SCORE_THRESHOLD       = 0.8
IOU_THRESHOLD         = 0.45
```

## Examples of predictions 

An example of correct drone detection in indoor environment shown below.

![Indoor Detection](/readme_images/indoor_prediction.jpg)

An example of correct drone detection in outdoor environment shown below.

![Outdoor Prediction](/readme_images/outdoor_prediction.jpg)

Video of model predictions shown in indoor environment shown below.

<video src="predicted_videos/model_2/free_2.mp4"></video>

