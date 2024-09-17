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
from geopy.distance import geodesic


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


def latlon2xy(latlon_point):
    # Homography matrix
    homography_matrix = [[-5.179012660893269, 55.151307185588124, 6983.035245667728304],
                         [10.366907722755846, -6.125985388589109, -1241.986418877135293],
                         [0.089249220866786, 0.042988199877047, 1.000000000000000]]
    
    # Convert latitude/longitude to pixel coordinates
    point_3D = [latlon_point[0], latlon_point[1], 1]  # Homogeneous coordinates (lat, lon, 1)
    pixel_coordinates = np.dot(homography_matrix, point_3D)  # Apply homography
    
    # Normalize by the scalar value
    scalar = pixel_coordinates[2]
    x = int(pixel_coordinates[0] / scalar)
    y = int(pixel_coordinates[1] / scalar)
    
    return (abs(x), abs(y))


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
    bearing_degrees = (360 + bearing_degrees) % 360

    return bearing_degrees


def calculate_speed(prev_center, current_center, time_diff):
## TO:DO: Needs to update
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    # distance = math.sqrt(dx**2 + dy**2)
    distance = haversine(current_center[1], current_center[0], prev_center[1], prev_center[0])
    # return distance / (1/13) if time_diff > 0 else 0
    return distance / time_diff if time_diff > 0 else 0


def apply_mask(frame):
    # Coordinates of the region of interest (detection zone)
    roi_vertices = [(414, 244), (539, 249), (761, 304), (829, 381), (661, 698), (161, 700), (117, 480), (198, 362)]
    mask = cv2.fillPoly(np.zeros_like(frame), np.array([roi_vertices], np.int32), (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

from datetime import datetime

def process_timestamp(time_str):
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    time_obj = datetime.strptime(time_str, time_format)
    timestamp_in_seconds = time_obj.timestamp()
    timestamp_in_seconds_with_milliseconds = round(timestamp_in_seconds, 3)
    return timestamp_in_seconds_with_milliseconds


# Helper function to predict future position based on current lat, lon, speed, and heading
def predict_position(lat, lon, speed, heading, time_step):
    """
    Predicts future latitude and longitude based on speed and heading.
    - lat, lon: current latitude and longitude (degrees)
    - speed: current speed in meters/second
    - heading: current heading in degrees
    - time_step: time step in seconds to predict future position
    Returns future (latitude, longitude).
    """
    # Convert heading to radians
    heading_rad = np.radians(heading)
    
    # Calculate distance traveled in time_step (in meters)
    distance_traveled = speed * time_step
    
    # Calculate delta latitude and delta longitude based on the heading
    delta_lat = (distance_traveled * np.cos(heading_rad)) / 111320  # Approx 111,320 meters per degree of latitude
    delta_lon = (distance_traveled * np.sin(heading_rad)) / (111320 * np.cos(np.radians(lat)))
    
    # New predicted position (latitude, longitude)
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    return new_lat, new_lon

# Function to compute Time-to-Collision (TTC) between two vehicles (circles)
def compute_ttc(lat1, lon1, lat2, lon2, speed1, speed2, heading1, heading2, radius, time_horizon=2, time_step=0.25):
    """
    Computes TTC by simulating future trajectories of two vehicles.
    - lat1, lon1: Vehicle 1's current latitude and longitude
    - lat2, lon2: Vehicle 2's current latitude and longitude
    - speed1, speed2: Speeds of the vehicles (in m/s)
    - heading1, heading2: Headings of the vehicles (in degrees)
    - radius: Collision threshold based on the sum of the vehicle's radii (in meters)
    - time_horizon: Maximum time horizon to simulate (seconds)
    - time_step: Time step for each simulation step (seconds)
    Returns the TTC if a collision is predicted, or infinity if no collision.
    """
    for t in np.arange(0, time_horizon, time_step):
        # Predict future positions of both vehicles
        future_lat1, future_lon1 = predict_position(lat1, lon1, speed1, heading1, t)
        future_lat2, future_lon2 = predict_position(lat2, lon2, speed2, heading2, t)
        
        # Calculate the distance between the two future positions
        distance = geodesic((future_lat1, future_lon1), (future_lat2, future_lon2)).meters
        
        # Check if distance is less than the collision radius (sum of the two vehicle's radii)
        if distance < 2 * radius:
            conflict_point = ((future_lat1 + future_lat2) / 2, (future_lon1 + future_lon2) / 2)
            return t, conflict_point  # Return the time to collision
    
    return float('inf'), None  # No collision predicted within the time horizon


# Function to detect potential conflicts using future trajectory prediction
def detect_conflict_with_prediction(vehicles, radius=1, ttc_threshold=0.5):
    """
    Detects potential conflicts between vehicles using linear trajectory prediction.
    - vehicles: List of vehicle data [(lat, lon, speed, heading), ...]
    - radius: Average radius of a vehicle in meters (assume circular vehicle shape)
    - ttc_threshold: TTC threshold for predicting a conflict (seconds)
    Returns a list of conflicts (vehicle index pairs and TTC).
    """
    conflicts = []
    key_list = list(vehicles.keys())
    for i in range(len(key_list)):
        for j in range(i + 1, len(key_list)):
            vehicle1 = vehicles[key_list[i]]
            vehicle2 = vehicles[key_list[j]]
            
            # Extract latitude, longitude, speed, and heading
            lat1, lon1, speed1, heading1 = vehicle1
            lat2, lon2, speed2, heading2 = vehicle2
            
            # Compute TTC between the two vehicles
            ttc, conflict_point = compute_ttc(lat1, lon1, lat2, lon2, speed1, speed2, heading1, heading2, radius)
            
            # Check if TTC is below the threshold
            if ttc < ttc_threshold and ttc > 0:
                conflicts.append((key_list[i], key_list[j], conflict_point, ttc))
    
    return conflicts


# def main(args):
def main(args, filename):
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


    # filename = "2024-09-12_16-13-10"

    fps = 0
    # input video
    # cap = cv2.VideoCapture(args.vid)
    cap = cv2.VideoCapture("../sample_video/test/output_video_" + filename + ".mp4")
    # input webcam
    # cap = cv2.VideoCapture(0)
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (video_width,video_height))
    out = cv2.VideoWriter('../sample_video/test/output/' + filename + '_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (video_width,video_height))

    path_save_results = "../sample_video/test/results"
    file_speed_w = open(path_save_results + "/" + filename + "_results.txt", "w")

    path_timestamp = "../sample_video/test"
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
        
        # current_time = time()

        row = df[df['Frame'] == counter]

        current_time_ori = row['Timestamp'].values[0]
        print(row['Frame'].values[0])
        

        current_time = process_timestamp(current_time_ori)

        print(counter)
        counter += 1
        

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
        vehicles = {}

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

                    file_speed_w.write(f"{class_name},{dx},{dy},{heading},{speed},{size},{float(t.score) * 100},{int(tid)},{timestamp}\n")

                    # if args.show:
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), get_random_color(tid), 2)
                    cv2.putText(frame, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    vehicles[tid] = (center[0], center[1], speed, heading)

            # Detect potential conflicts
            conflicts = detect_conflict_with_prediction(vehicles)

            # Output conflicts
            if conflicts:
                print(f"Potential conflicts detected: {conflicts}")
                # (key_list[i], key_list[j], conflict_point, ttc)
                for conflict in conflicts:
                    print(latlon2xy(conflict[2]))
                    vehicle_1 = latlon2xy(prev_centers[conflict[0]])
                    vehicle_2 = latlon2xy(prev_centers[conflict[1]])
                    conflict_point = latlon2xy(conflict[2])
                    vehicle_1_pred = predict_position(prev_centers[conflict[0]][0], prev_centers[conflict[0]][1], vehicles[conflict[0]][2], vehicles[conflict[0]][3], 0.5)
                    vehicle_2_pred = predict_position(prev_centers[conflict[1]][0], prev_centers[conflict[1]][1], vehicles[conflict[1]][2], vehicles[conflict[1]][3], 0.5)


                    cv2.circle(frame, conflict_point, radius=5, color=(0, 0, 255), thickness=1)
                    cv2.circle(frame, vehicle_1, radius=5, color=(0, 255, 0), thickness=1)
                    cv2.circle(frame, vehicle_2, radius=5, color=(0, 255, 0), thickness=1)
                    cv2.line(frame, vehicle_1, latlon2xy(vehicle_1_pred), color=(0, 255, 0), thickness=1)
                    cv2.line(frame, vehicle_2, latlon2xy(vehicle_2_pred), color=(0, 255, 0), thickness=1)
                    
                output_filename = f"./conflict/conflict_{current_time_ori}.jpg"
                cv2.imwrite(output_filename, frame)
            else:
                print("No conflicts detected.")
                
        end = float(time())
        
        
        fps = 1/(end - start)
        print(fps_str)
        cv2.putText(frame, "YOLOV8-BYTETrack", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(frame, fps_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if args.show:
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    # tracker_trt.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    parser.add_argument('--vid', type=str, help='Video file', default='../sample_video/2024_08_28_11_29_00_raw.mp4')
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

    list_filename = [
 '2024-09-12_16-17-53',
 '2024-09-12_16-18-40',
 '2024-09-12_16-21-24',
 '2024-09-12_16-22-03',
 '2024-09-12_16-24-56',
 '2024-09-12_16-28-14',
 '2024-09-12_16-30-37',
 '2024-09-12_16-32-47',
 '2024-09-12_16-36-18',
 '2024-09-12_16-37-50',
 '2024-09-12_16-42-05',
 '2024-09-12_16-45-00',
 '2024-09-12_16-50-45',
 '2024-09-12_16-53-03',
 '2024-09-12_17-01-23',
 '2024-09-12_17-07-25']
    
    for i in range(len(list_filename)):
        main(args, list_filename[i])
