from models import TRTModule
import argparse
from time import time
import cv2
from pathlib import Path
import torch
import ctypes
import tracker_trt


from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from datetime import datetime, timedelta
import json

output_file = "tracking_output.txt"


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


def write_data_to_file(data, filename):
    try:
        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")
    except Exception as e:
        print("Error writing data to file:", e)
        
def draw_roi(frame):
    for key, value in DICT_ROIS.items():
        cv2.rectangle(frame, (value.x1, value.y1), (value.x2, value.y2), (0, 0, 255), 1)
        cv2.putText(frame, key, (value.x1, value.y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
    return frame

def determine_region(cx, cy):
    for key, value in DICT_ROIS.items():
        if cx >= value.x1 and cx <= value.x2 and cy >= value.y1 and cy <= value.y2:
            return key
    return None



def main(args):
    
    
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    fps = 0
    # input video
    cap = cv2.VideoCapture(args.vid)
    # input webcam
    # cap = cv2.VideoCapture(0)
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    while(True):
        ret, frame = cap.read()
        
        if frame is None:
            print('No image input!')
            break
        
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
        detections = []
        for (bbox, score, label) in zip(bboxes, scores, labels):
            if label == 0 and score.item() > 0.5:
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                detections.append((bbox[0], bbox[1], bbox[2] , bbox[3], cls, score.item()))
        end = float(time())
        
        list_bbox = tracker_trt.update(detections,frame)
        for (x1, y1, x2, y2, cls, track_id) in list_bbox:
            color = [0, 255, 0]
                
            if args.show:
                frame = draw_roi(frame)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{cls} {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

    
        fps = 1/(end - start)
        print(fps_str)
         
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if args.show:
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        # out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    # tracker_trt.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    parser.add_argument('--vid', type=str, help='Video file', default='../sample_video/sample.mp4')
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

