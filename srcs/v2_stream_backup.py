from models import TRTModule
import argparse
from time import time
import cv2
from pathlib import Path
import torch
import ctypes
from bytetrack.dev_byte_tracker import BYTETracker

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from datetime import datetime, timedelta
import json
import numpy as np
import random
from math import radians, cos, sin, asin, sqrt
import math
from pyproj import Transformer
import csv
from datetime import datetime

import threading
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()
tracked_objects = []

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


def calculate_center_point(bbox):
## TO:DO: Needs to update
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def xy2latlon(point):
    #homography_matrix
    homography_matrix_obtained = [[-5.179012660893269, 55.151307185588124, 6983.035245667728304],
                                  [10.366907722755846, -6.125985388589109, -1241.986418877135293],
                                  [0.089249220866786, 0.042988199877047, 1.000000000000000]]
    mat = homography_matrix_obtained
    matinv = np.linalg.inv(mat)#.I
    # Convert to lat/lon
    point_3D = [point[0], point[1], 1]
    hh = np.dot(matinv,point_3D)
    scalar = hh[2]
    latitude = hh[0]/scalar
    longitude = hh[1]/scalar
    return (latitude, longitude)


# Function to calculate the great circle distance between two points on Earth
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees).
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r * 1000  # Convert to meters


def calculate_bearing(prev, curr):
    """
    Calculate the bearing angle from point (lat1, lon1) to point (lat2, lon2)
    North is 0 degrees, increasing clockwise.
    Latitude and longitude should be in radians.
    """
    # Convert latitude and longitude from degrees to radians
    lat1 = prev[0]; lon1 = prev[1]; lat2 = curr[0]; lon2 = curr[1]
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Calculate the difference in longitude
    delta_lon = lon2 - lon1

    # Calculate bearing angle
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    # Use atan2 to compute the bearing
    bearing_radians = math.atan2(x, y)

    # Convert the bearing to degrees
    bearing_degrees = math.degrees(bearing_radians)

    # Normalize the bearing to a 0-360 degree range
    bearing_degrees = (bearing_degrees + 360) % 360

    return bearing_degrees


def calculate_speed(prev_center, current_center, time_diff):
## TO:DO: Needs to update
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    # distance = math.sqrt(dx**2 + dy**2)
    distance = haversine(current_center[1], current_center[0], prev_center[1], prev_center[0])
    # return distance / (1/13) if time_diff > 0 else 0
    return distance / time_diff if time_diff > 0 else 0


def process_timestamp(time_str):
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    time_obj = datetime.strptime(time_str, time_format)
    timestamp_in_seconds = time_obj.timestamp()
    timestamp_in_seconds_with_milliseconds = round(timestamp_in_seconds, 3)
    return timestamp_in_seconds_with_milliseconds


# def main(args):
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

    fps = 0
    # input video
    cap = cv2.VideoCapture(args.vid)


    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.udp_save:
        path_save_results = args.result_dir+ 'results'
        file_speed_w = open(path_save_results + "/" + filename + "_results.txt", "w")

    if args.recorded_time:
        path_timestamp = args.result_dir
        filename_timestamp = "frame_timestamps_" + filename + ".csv"

        import pandas as pd
        file_timestamp = path_timestamp + "/" + filename_timestamp
        df = pd.read_csv(file_timestamp)


    roi_vertices = np.array([[414, 244], [539, 249], [761, 304], [829, 381], [661, 698], [161, 700], [117, 480], [198, 362]], np.int32)

    prev_centers = {}
    prev_times = {}

    counter = 0

    while(True):
        ret, frame = cap.read()

        if frame is None:
            print('No image input!')
            break


        if args.recorded_time:
            row = df[df['Frame'] == counter]
            current_time_ori = row['Timestamp'].values[0]
            current_time = process_timestamp(current_time_ori)
            counter += 1
        else:
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

        output = []

        for (bbox, score, label) in zip(bboxes, scores, labels):
            if score.item() > 0.2:
                bbox = bbox.round().int().tolist()
                cls_id = int(label)

                cls = CLASSES[cls_id]
                # x1, y1, x2, y2, conf
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], score.item(), cls_id]) #10

        output = np.array(output)

        info_imgs = frame.shape[:2]
        img_size = info_imgs

        if output != []:
            online_targets = tracker.update(output, info_imgs, img_size)
            tracked_objects.clear()
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

                clss = t.cls
                class_name = CLASSES[int(clss)]
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_cls.append(class_name)

                center_2D = calculate_center_point(bbox)
                center = xy2latlon(center_2D)

                if tid in prev_centers and tid in prev_times:
                    heading = calculate_bearing(prev_centers[tid], center)
                    speed = calculate_speed(prev_centers[tid], center, current_time - prev_times[tid])
                else:
                    heading = 0
                    speed = 0

                size = ""
                if int(clss) in [0, 1, 3]:
                    size = 'small'
                elif int(clss) in [2]:
                    size = 'medium'
                elif int(clss) in [5, 7]:
                    size = 'large'
                else:
                    size = 'N/A'

                list_tem = str(current_time).split(".")
                list_tem[1] = list_tem[1][0:3]
                timestamp = str(list_tem[0] + '.' + list_tem[1])

                lat_origin, lon_origin = 47.6277146, -122.1432525
                lat_target, lon_target = center[0], center[1]
                transformer = Transformer.from_crs("epsg:4326", "epsg:32610")
                x_origin, y_origin = transformer.transform(lat_origin, lon_origin)
                x_target, y_target = transformer.transform(lat_target, lon_target)

                dx = str(x_target - x_origin)  # "X difference (East): {dx} meters"
                dy = str(y_target - y_origin)  # "Y difference (North): {dy} meters"

                prev_centers[tid] = center
                prev_times[tid] = current_time

                if cv2.pointPolygonTest(roi_vertices, center_2D, False) >= 0:
                    if args.udp_save:
                        file_speed_w.write(f"{class_name},{dx},{dy},{heading},{speed},{size},{float(t.score) * 100},{int(tid)},{timestamp}\n")

                    # if args.show:
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), get_random_color(tid), 2)
                    cv2.putText(frame, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    tracked_objects.append({
                    'track_id': tid,
                    'class_name': class_name,
                    'speed': speed,
                    'heading': heading,
                    'size': size,
                    'score': float(t.score)*100,
                    })


        # out.write(frame)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    parser.add_argument('--vid', type=str, help='Video files', default='../sample_video/2024_08_28_11_29_00_raw.mp4')
    # parser.add_argument('--vid', type=str, help='Video file', default='rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047NE.stream')
    parser.add_argument('--device', type=str, default='cuda:0', help='TensorRT infer device')
    parser.add_argument('--udp_save', action='store_true', help='Save the 9 outputs in txt files')
    parser.add_argument('--recorded_time', action='store_true', help='Use time when the image was recored. From csv file.')
    parser.add_argument('--result_dir', type=str, help='UDP save directory', default='../sample_video/test')

    args = parser.parse_args()

    threading.Thread(target=main, args=(args,), daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
