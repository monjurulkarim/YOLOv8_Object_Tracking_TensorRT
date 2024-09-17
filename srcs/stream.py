import cv2
import numpy as np
import threading
from flask import Flask, Response, render_template, jsonify
from models import TRTModule
import argparse
from time import time
import torch
from bytetrack.dev_byte_tracker import BYTETracker
from config import CLASSES
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox
import random

app = Flask(__name__)

# Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()
tracked_objects = []

# CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#            'scissors', 'teddy bear', 'hair drier', 'toothbrush')

color_dict = {}

def get_random_color(id):
    if id not in color_dict:
        color_dict[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_dict[id]

def main(args):
    global latest_frame

    args_bytetrack = argparse.Namespace()
    args_bytetrack.track_thresh = 0.2
    args_bytetrack.track_buffer = 200
    args_bytetrack.mot20 = True
    args_bytetrack.match_thresh = 0.7

    tracker = BYTETracker(args_bytetrack)
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    cap = cv2.VideoCapture(args.vid)

    while True:
        ret, frame = cap.read()

        if frame is None:
            print('No image input!')
            break

        bgr, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)

        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)

        if bboxes.numel() > 0:
            bboxes -= dwdh
            bboxes /= ratio

            output = []
            for (bbox, score, label) in zip(bboxes, scores, labels):
                if score.item() > 0.2:
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    output.append([bbox[0], bbox[1], bbox[2], bbox[3], score.item(), cls_id])

            output = np.array(output)

            if len(output) > 0:
                online_targets = tracker.update(output, frame.shape[:2], frame.shape[:2])
                tracked_objects.clear()

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

                    clss = t.cls
                    class_name = CLASSES[int(clss)]
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])),
                                  (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                                  get_random_color(tid), 2)
                    cv2.putText(frame, str(tid), (int(tlwh[0]), int(tlwh[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    tracked_objects.append({
                        'track_id': tid,
                        'class_name': class_name,
                        'bbox': bbox
                    })



        with frame_lock:
            latest_frame = frame.copy()

    cap.release()

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/tracked_objects')
def get_tracked_objects():
    global tracked_objects
    return jsonify(tracked_objects)

@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/')
# def index():
#     global tracked_objects
#     return render_template('index.html', tracked_objects=tracked_objects)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    parser.add_argument('--vid', type=str, help='Video file', default='../sample_video/2024_08_28_11_29_00_raw.mp4')
    parser.add_argument('--device', type=str, default='cuda:0', help='TensorRT infer device')
    args = parser.parse_args()

    threading.Thread(target=main, args=(args,), daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
