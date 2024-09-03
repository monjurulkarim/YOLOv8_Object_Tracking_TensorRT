from models import TRTModule
import argparse
from time import time
import cv2
from pathlib import Path
import torch
import ctypes
from bytetrack.byte_tracker import BYTETracker

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from datetime import datetime, timedelta
import json
import numpy as np
import random
import socket
import json
import math
# import time



# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def calculate_center_point(bbox):
## TO:DO: Needs to update
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_heading(prev_center, current_center):
## TO:DO: Needs to update
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    return math.degrees(math.atan2(dy, dx)) % 360

def calculate_speed(prev_center, current_center, time_diff):
## TO:DO: Needs to update
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance / time_diff if time_diff > 0 else 0

def calculate_size(bbox):
## This is just for testing udp data transfer. TO:DO: Needs to update
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if area < 5000:
        return "Small"
    elif area < 15000:
        return "Medium"
    else:
        return "Large"

class ROI:
    def __init__(self, x1, y1, x2, y2, roi_id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.roi_id = roi_id
        self.count = 0
        

DICT_ROIS = {}
DEBOUNCE_PERIOD = timedelta(seconds=2)
person_tracker = {}
debounce_tracker = {}

color_dict = {}

def get_random_color(id):
    if id not in color_dict:
        color_dict[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_dict[id]



def main(args):
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

    # Set up UDP socket
    UDP_IP = "128.95.204.54"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    fps = 0
    # input video
    cap = cv2.VideoCapture(args.vid)
    # input webcam
    # cap = cv2.VideoCapture(0)
    
    # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (video_width,video_height))

    prev_centers = {}
    prev_times = {}
    # class_ids = {}
    class_names = {}

    while(True):
        ret, frame = cap.read()
        
        if frame is None:
            print('No image input!')
            continue

        current_time = time()
        
        start = float(time())
        fps_str = "FPS:"
        fps_str += "{:.2f}".format(fps)
        bgr = frame
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        tensor = blob(rgb, return_seg=False)
        
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        
        tensor = torch.asarray(tensor, device=device)
        
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        # print(labels)
        
        if bboxes.numel() == 0:
            continue
        
        bboxes -= dwdh
        bboxes /= ratio
        tracker_input = np.concatenate([bboxes.cpu().numpy(), scores.cpu().numpy()[:, None]], axis=1)
        online_targets = tracker.update(tracker_input, [H, W], [H, W])

        frame_data = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            center = calculate_center_point(bbox)
            
            if tid not in class_names:
                class_index = int(labels[len(class_names)].item())
                class_names[tid] = CLASSES[class_index] if class_index < len(CLASSES) else "Unknown"

            if tid in prev_centers and tid in prev_times:
                heading = calculate_heading(prev_centers[tid], center)
                speed = calculate_speed(prev_centers[tid], center, current_time - prev_times[tid])
            else:
                heading = 0
                speed = 0

            size = calculate_size(bbox)
            # class_name = CLASSES.get(class_ids[tid], "Unknown")
            # class_name = CLASSES.get(class_ids[tid], "Unknown")

            ## Test: UDP data 
            frame_data.append({
                "Class": class_names[tid],
                "x": center[0],
                "y": center[1],
                "Heading": heading,
                "Speed": speed,
                "Size": size,
                "Confidence": float(t.score) * 100,  # Convert to percentage
                "TrackID": int(tid),
                "Timestamp": int(current_time)
            })

            prev_centers[tid] = center
            prev_times[tid] = current_time
            
            if args.show:
                cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), 
                              (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), 
                              get_random_color(tid), 2)
                cv2.putText(frame, f"{class_names[tid]}-{tid}", (int(tlwh[0]), int(tlwh[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"{class_name}-{tid}", (int(tlwh[0]), int(tlwh[1])), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Serialize and send data over UDP
        json_data = json.dumps(frame_data)
        sock.sendto(json_data.encode(), (UDP_IP, UDP_PORT))
        
        end = float(time())     
        fps = 1/(end - start)
        
        if args.show:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 ByteTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    #parser.add_argument('--vid', type=str, help='Video file', default='../sample_video/001.mp4')
    parser.add_argument('--vid', type=str, help='Video file', default='rtsp://65.76.54.158:554/1/h264major')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the results')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

