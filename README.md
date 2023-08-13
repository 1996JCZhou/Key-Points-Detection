# Keypoint Detection using YOLOv8
Keypoint Detection stands as a fundamental and core task in the realm of computer vision. This task involves the identification of semantically significant points within an image, ultimately yielding the coordinates of these points. The complete process of keypoint detection encompasses various stages: dataset annotation, deep learning frameworks of [YOLOv8](https://github.com/ultralytics/ultralytics) training, performance evaluation, inference and prediction, as well as application deployment. Whether you're a novice or an experienced computer vision enthusiast, this README provides insights into the key aspects of the keypoint detection pipeline. From understanding the importance of dataset annotation to mastering the training of deep learning models, this repository aims to guide you through each stage of the keypoint detection journey. Feel free to explore the provided resources, codes, and documentation. :)

## Requirements
- python
- opencv-python
- json
- matplotlib
- os
- shutil
- random
- seedir
- tqdm

## Data preparation
For data preparation the first step is to label your own key point detection dataset. I used the labeling tool [Labelme](https://github.com/wkentaro/labelme) to label boxes, points, and polylines to form a **keypoint detection dataset for set squares with 30-60-90 degree angles**. Furthermore, interpret the labelme annotation file and use [OpenCV](https://github.com/opencv/opencv) in python scripts to visualize the annotation information. Lay the foundation for subsequent label format conversion and algorithm training.

### Step 1
Down [Labelme](https://github.com/wkentaro/labelme) for windows on this [page](https://github.com/wkentaro/labelme/releases/tag/v5.3.0). Open Labelme directly without installation and choose an image to annotate.

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/Labelme%201.PNG)

![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/Labelme%202.PNG)

### Setp 2
Visualize the annotation information using the [annotationVisualization](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/annotationVisualization.py) script in python.
![image](https://github.com/1996JCZhou/Key-Points-Detection/blob/master/Images%20for%20README/annotationInformation.png)



