This repository contains the code developed by STAR Lab for object detection, tracking, and conflict point estimation on an edge device using live video streams. We utilized OpenCV to capture video from various sources, including cameras, video files, or RTSP streams, and implemented **YOLOv8 TensorRT** for object detection, along with **BYTETrack** for object tracking. Key features of this codebase include:
- Object detection
- Object tracking
- Calculation of speed, heading, and distance
- Conflict point estimation and time-to-collision (TTC) calculation
- Output transmission via UDP protocol
- Hosting a web app for visualizing results through a web interface

The edge device used for this project is the NVIDIA JETSON ORIN NANO. Note: Different versions of MUST sensors are sold with various JETSON devices. Please ensure your MUST sensor includes this platform or a more advanced one.
## Getting Started

### Environment

- NVIDIA CUDA: 12.2
- NVIDIA TensorRT >= 8.6.2


#### Clone repository

Clone repository and submodules

```bash
git clone --recurse-submodules https://github.com/monjurulkarim/YOLOv8_Object_Tracking_TensorRT.git
```

#### Prepare enviroment

Create new enviroment

```bash
conda create -n yolov8_ds python=3.8
```

Activate enviroment

```bash
conda activate yolov8_ds
```

### Prepare models
Go to **`refs/YOLOv8-TensorRT`** and install requirements for exporting models

```bash
cd refs/YOLOv8-TensorRT
pip3 install -r requirements.txt
pip3 install tensorrt easydict pycuda lap cython_bbox
pip3 install flask
```
Install `python3-libnvinfer`

```bash
sudo apt-get install python3-libnvinfer
```

## (Optional steps)
The following steps (1-3) are optional because we have already shared our converted weights which are ready to be used. You don't need to convert the weights.

1. Save the YOLOv8 weights in folder **`models/to_export`**

2. **Export YOLOv8 ONNX model**

In **`refs/YOLOv8-TensorRT`** run the following command to export YOLOv8 ONNX model

```bash
python3 export-det.py \
--weights ../../models/to_export/yolov8n.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

The output `.onnx` model will be saved in **`models/to_export`** folder, move the model to **`models/onnx`** folder
```bash
mv ../../models/to_export/yolov8n.onnx ../../models/onnx/yolov8n.onnx
```
3. **Export YOLOv8 TensorRT model**

In **`refs/YOLOv8-TensorRT`** run the following command to export YOLOv8 TensorRT model

```bash
python3 build.py \
--weights ../../models/onnx/yolov8n.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
```
The output `.engine` model will be saved in **`models/onnx`** folder, move the model to **`models/trt`** folder

```bash
mv ../../models/onnx/yolov8n.engine ../../models/engine/yolov8n.engine
```

**Build OpenCV**

```bash
bash build_opencv.sh
```

## Some code description

`srcs/yolov8_bytetrack_starlab.py`: This is the main script responsible for running and generating all required results. Users need to specify different parser arguments using flags (e.g., `--show`, `--vid`). Running this script starts the web app, allowing users to visualize the results in a browser within the local network.

`srcs/template/index.html`: This file contains the HTML structure of the web app used for visualization.

`srcs/config.py`: contains different parameters

`models/` : contains the weights

For more details we added additional comments in the code.
### Run script

```

**Go to `srcs` folder**

```bash
cd srcs
```


**Run YOLOv8 + BYTETrack**

```bash
python3 yolov8_bytetrack_starlab.py

```

Running this code will start an web app which can be accessed using any browser. Please make sure to define your devices IP address in the `templates/index.html` at line # 71.

---

# References

- [ultralytics](https://github.com/ultralytics/ultralytics)
- [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
