# Keypoint Detection using YOLOv8 in 6 Steps
Keypoint Detection stands as a fundamental and core task in the realm of computer vision. This task involves the identification of semantically significant points within an image, ultimately yielding the coordinates of these points. The complete process of keypoint detection encompasses various stages: dataset annotation, deep learning frameworks of [YOLOv8](https://github.com/ultralytics/ultralytics) training, performance evaluation, inference and prediction, as well as application deployment. Whether you're a novice or an experienced computer vision enthusiast, this README provides insights into the key aspects of the keypoint detection pipeline. From understanding the importance of dataset annotation to mastering the training of deep learning models, this repository aims to guide you through each stage of the keypoint detection journey. Feel free to explore the provided resources, codes, and documentation. :)

## Requirements
- python
- opencv-python
- json
- matplotlib
- os
- shutil
- random
- seedir emoji
- tqdm
- ultralytics
- numpy
- pillow
- seaborn
- pandas
- wandb

## Data preparation
For data preparation the first step is to label your own key point detection dataset. I used the labeling tool [Labelme](https://github.com/wkentaro/labelme) to label boxes, points, and polylines to form a **keypoint detection dataset for set squares with 30-60-90 degree angles**. Furthermore, interpret the labelme annotation file and use [OpenCV](https://github.com/opencv/opencv) in python scripts to visualize the annotation information. Lay the foundation for subsequent label format conversion and algorithm training.

### Step 1 - Dataset annotation
Down [Labelme](https://github.com/wkentaro/labelme) for windows on this [page](https://github.com/wkentaro/labelme/releases/tag/v5.3.0). Open Labelme directly without installation and choose an image to annotate.

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/Labelme%201.PNG)

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/Labelme%202.PNG)

Visualize the annotation information using the [annotationVisualization](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/annotationVisualization.py) script in python.
![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/annotationInformation.png)

### Step 2 - Dataset split
After you have collected enough images to form your own dataset and annotated all of them, you will get a dataset, which looks like [this](https://github.com/1996JCZhou/Key-Points-Detection/tree/master/Setsquare_Keypoint_Labelme). Then you need to split your dataset for training and validation by using the [dataSplit](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/dataSplit.py) script. In the [const]() script you need to define the address of your source dataset, the address to save your taining and validation datasets and the rate to randomly split images for training and validation.

After using the [seedir](https://github.com/earnestt1234/seedir) package, the structure will look like this:

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/structure.PNG)

### Step 3 - Annotation information transformation
Use the [labelme2YOLO](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/labelme2YOLO.py) script to transform the json files for image annotation information into txt files that we can feed the YOLOv8 frameworks directly. Center coordinates, width and height of all bounding boxes along with coordinate of each keypoint are normalized by the width or height of the image. After transformation, you will get a dataset, which looks like [this](https://github.com/1996JCZhou/Key-Points-Detection/tree/master/KeyPointDetection_YOLO).

The contents in a txt file will look like this:
![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/Annotation%20information.png)

You are now for training the YOLOv8 framework using your own dataset!

## Training of deep learning frameworks of YOLOv8

### Step 4 - YOLOv8 installation
pip install ultralytics --upgrade

To verify if the installation was successful, type "python" in the command line interface to start the python environment. Then "import ultralytics", followed by "ultralytics.checks()". If the installation was successful, you will see this:
![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/ultralytics.PNG)

### Step 5 - Cloud Computing Platform Setup
Since I don't have a high performance GPU of my own, I chose the [Featurize](https://featurize.cn/) as my Cloud Computing Platform. There are many types of high performance GPUs out there for you to choose from.

### Step 6 - Training of YOLOv8 using Transfer Learning on a single GPU
Before training, we need to define a yaml file, which contains information about the datasets for taining and validation, keypoints and the category of the bounding box. An example of the yaml file looks like [this](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/setSquare_KD_YOLO.yaml).

Here is the [template](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/YOLOv8%20command%20template.PNG) of YOLOv8 commands. I have used several official YOLOV8 keypoint detection pre-training models to train my **keypoint detection dataset for set squares with 30-60-90 degree angles**. They are: **yolov8n-pose.pt**, **yolov8s-pose.pt**, **yolov8m-pose.pt**, **yolov8l-pose.pt**, **yolov8x-pose.pt** and **yolov8x-pose-p6.pt**. You can search for more details about arguments for training process on this official [page](https://docs.ultralytics.com/modes/train/).

Type one of the following commands in the command line interface to start training:
- yolo pose train data=Triangle_215.yaml model=yolov8n-pose.pt pretrained=True project=setSquare_KD name=n_pretrain epochs=50 batch=16 device=0
- yolo pose train data=Triangle_215.yaml model=yolov8s-pose.pt pretrained=True project=setSquare_KD name=s_pretrain epochs=50 batch=16 device=0
- yolo pose train data=Triangle_215.yaml model=yolov8m-pose.pt pretrained=True project=setSquare_KD name=m_pretrain epochs=50 batch=16 device=0
- yolo pose train data=Triangle_215.yaml model=yolov8l-pose.pt pretrained=True project=setSquare_KD name=l_pretrain epochs=50 batch=4 device=0
- yolo pose train data=Triangle_215.yaml model=yolov8x-pose.pt pretrained=True project=setSquare_KD name=x_pretrain epochs=50 batch=4 device=0
- yolo pose train data=Triangle_215.yaml model=yolov8x-pose-p6.pt pretrained=True imgsz=1280 project=setSquare_KD name=x_p6_pretrain epochs=50 batch=2 device=0

## Model inference and prediction
### Inference and prediction of a single image
yolo pose predict model=TRAINED_MODEL.pt source=IMAGE_PRED.jpg device=0

### Inference and prediction of a video
yolo pose predict model=TRAINED_MODEL.pt source=VIDEO_PRED.mp4 device=0 verbose=False

### Inference and prediction of real-time camera images
yolo pose predict model=TRAINED_MODEL.pt source=0 show verbose=False

## My own results
### Inference and prediction of a single image

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/output_image.jpg)

### Inference and prediction of a video
Welcome to see my videos in my Youtube channel for this project.

A short video for quick test: https://www.youtube.com/shorts/VOS-zydVQg8.

Original video: https://www.youtube.com/watch?v=PpCS5-kpklY,

Result video after training 100 Epochs: https://www.youtube.com/watch?v=oKym5_25l40 and

Result video after training 300 Epochs: https://www.youtube.com/watch?v=-ToolcpQbsI.

