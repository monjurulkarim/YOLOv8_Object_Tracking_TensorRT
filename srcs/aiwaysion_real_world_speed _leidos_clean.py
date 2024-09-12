import numpy as np
import cv2
import argparse
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from datetime import datetime, timedelta, timezone
from pyproj import Transformer

# Command line argument parsing
parser = argparse.ArgumentParser(description="Real world speed")
parser.add_argument("filename", type=str, help="Filename")
parser.add_argument("path", type=str, help="Path")

# Parse arguments
args = parser.parse_args()
filename = args.filename
path = args.path

# Calculate the direction angle between two points
def calculate_direction_angle(x1, y1, x2, y2):
    # Calculate displacement vector
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    # Use atan2 to compute the direction angle
    angle_radians = math.atan2(delta_y, delta_x)
    
    # Convert the angle to degrees if needed
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

####################################################################################################

# Calculate the bearing (direction) between two geographical points
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing angle from point (lat1, lon1) to point (lat2, lon2)
    North is 0 degrees, increasing clockwise.
    Latitude and longitude should be in radians.
    """
    # Convert latitude and longitude from degrees to radians
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

###################################################################################################

# Function to convert PDT time to Unix timestamp
def pdt_to_unix(pdt_time_str):
    # Parse the PDT time string
    pdt_time = datetime.strptime(pdt_time_str, "%Y_%m_%d_%H_%M_%S.%f")
    # PDT is UTC-7 during daylight saving time
    utc_time = pdt_time + timedelta(hours=7)
    # Get Unix timestamp
    unix_timestamp = int(utc_time.replace(tzinfo=timezone.utc).timestamp())
    return unix_timestamp


homography_matrix_obtained = [[-5.179012660893269, 55.151307185588124, 6983.035245667728304],
                              [10.366907722755846, -6.125985388589109, -1241.986418877135293],
                              [0.089249220866786, 0.042988199877047, 1.000000000000000]]

# Invert the homography matrix
mat = homography_matrix_obtained
matinv = np.linalg.inv(mat)

file = open(path + "/" + filename + ".txt")
file_w = open(path + "/" + filename + "_results_converted_sort.txt", "w")

track_map = {}
for line in file:
    s = line.split(" ")
    if s[2] in track_map:
        track_map[s[2]].append(line)
    else:
        track_map[s[2]] = [line]

# Process tracking points and apply homography transformation
for trk in track_map:
    for i in range(len(track_map[trk])):
        s = track_map[trk][i].split(" ")
        x = int(float(s[3]))
        y = int(float(s[4]))
        w = int(float(s[5]))
        h = int(float(s[6]))
        c_x = x + w / 2
        c_y = y + h / 2
        point = [c_x, c_y, 1]
        hh = np.dot(matinv, point)
        scalar = hh[2]
        file_w.write(track_map[trk][i].replace("-1 -1 -1 -1", str(hh[0] / scalar) + " " + str(hh[1] / scalar)))

track_speed = {}
file_3d = open(path + "/" + filename + "_results_converted_sort.txt")
for line in file_3d:
    s = line.split(" ")
    cid = s[2]
    if cid in track_speed:
        track_speed[cid].append(line)
    else:
        track_speed[cid] = [line]

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

# Variables to store speed data
track_speed_km = {}
track_speed_mile = {}

fps = 14
for trk in track_speed:
    trk_len = len(track_speed[trk])
    speed_list = []
    for i in range(trk_len - 1):
        s1 = track_speed[trk][i].replace(" \n", "").split(" ")
        s2 = track_speed[trk][i + 1].replace(" \n", "").split(" ")
        c3d_x1 = float(s1[-2])
        c3d_y1 = float(s1[-1])
        c3d_x2 = float(s2[-2])
        c3d_y2 = float(s2[-1])
        distance = haversine(c3d_y1, c3d_x1, c3d_y2, c3d_x2)
        speed = distance / (1 / fps)
        speed_list.append(speed)

    x = np.arange(trk_len - 1)
    y_must_speed = np.array(speed_list)

    if len(x) > 3:  # Check for cubic splines
        tck = splrep(x, y_must_speed, s=1000)  # Adjust smoothness with 's' parameter
        x_new = np.linspace(0, len(x), 200)
        y_smooth = splev(x_new, tck)
        y_smooth_original_x = splev(x, tck)
    else:
        y_smooth_original_x = y_must_speed

    for i in range(trk_len):
        s1 = track_speed[trk][i].replace(" \n", "").split(" ")
        s2 = track_speed[trk][-1].replace(" \n", "").split(" ")
        c3d_x1 = float(s1[-2])
        c3d_y1 = float(s1[-1])
        c3d_x2 = float(s2[-2])
        c3d_y2 = float(s2[-1])
        distance = haversine(c3d_y1, c3d_x1, c3d_y2, c3d_x2)
        frameId1 = int(s1[0])
        frameId2 = int(s2[0])
        time_diff = float(frameId2 - frameId1) / fps

        direction_angle = calculate_bearing(c3d_x1, c3d_y1, c3d_x2, c3d_y2)
        print(f"Direction angle: {direction_angle:.9f} degrees")

        if time_diff > 0:
            speed = y_smooth_original_x[i]
            if trk in track_speed_km:
                track_speed_km[trk].append([speed, direction_angle])
                track_speed_mile[trk].append([speed, direction_angle])
            else:
                track_speed_km[trk] = [[speed, direction_angle]]
                track_speed_mile[trk] = [[speed, direction_angle]]
        else:
            if trk in track_speed_km:
                track_speed_km[trk].append([0.0, 0.0])
                track_speed_mile[trk].append([0.0, 0.0])
            else:
                track_speed_km[trk] = [[0.0, 0.0]]
                track_speed_mile[trk] = [[0.0, 0.0]]

file_speed_w = open(path + "/" + filename + "_results_converted_sort_speed.txt", "w")
for trk in track_speed:
    trk_len = len(track_speed[trk])
    for i in range(trk_len):
        fID = track_speed[trk][i].split(" ")[0]

        fps = 14
        frameid = int(fID)

        ss = filename.split("_")
        ss_year = ss[0]
        ss_month = ss[1]
        ss_day = ss[2]
        ss_hr = int(ss[3])
        ss_min = int(ss[4])
        ss_sec = int(ss[5])

        sec = frameid / fps + ss_sec
        min = int(sec / 60) + ss_min
        sec = int(sec % 60)
        msec = (frameid % fps) * 0.0714
        msec = str(round(msec, 2))[1:]

        if min >= 60:
            min -= 60
            ss_hr += 1
        if ss_hr >= 24:
            ss_hr -= 24

        ss_hr = f"{ss_hr:02d}"
        min = f"{min:02d}"
        sec = f"{sec:02d}"

        new_timestamp = f"{ss_year}_{ss_month}_{ss_day}_{ss_hr}_{min}_{sec}{msec}"

        new_timestamp = pdt_to_unix(new_timestamp)

        result = str(new_timestamp) + " " + track_speed[trk][i].replace("\n", str(track_speed_mile[trk][i - 1][0]) + " " + str(track_speed_mile[trk][i - 1][1]))
        output = result.split(" ")

        obj_cls = output[2]
        size = ""
        time = output[0]

        if output[2] == "2":
            size = 'small'
        elif output[2] == "3":
            size = 'medium'
        elif output[2] == "7":
            size = 'large'
        else:
            size = 'N/A'

        x = int(float(output[4]))
        y = int(float(output[5]))
        w = int(float(output[6]))
        h = int(float(output[7]))
        c_x = str(x + w / 2)
        c_y = str(y + h / 2)

        conf = output[8]
        lat = output[9]
        lon = output[10]
        speed = output[11]
        heading = output[12]
        car_id = output[3]

        lat_origin, lon_origin = 47.6277146, -122.1432525
        lat_target, lon_target = float(lat), float(lon)
        transformer = Transformer.from_crs("epsg:4326", "epsg:32610")
        x_origin, y_origin = transformer.transform(lat_origin, lon_origin)
        x_target, y_target = transformer.transform(lat_target, lon_target)

        dx = str(x_target - x_origin)
        dy = str(y_target - y_origin)

        print(f"X difference (East): {dx} meters")
        print(f"Y difference (North): {dy} meters")

        car_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
        car_cls = car_classes[int(obj_cls)]

        file_speed_w.write(f"{car_cls},{dx},{dy},{heading},{speed},{size},{conf},{car_id},{time}\n")
