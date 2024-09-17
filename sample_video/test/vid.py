import cv2
import csv
from datetime import datetime
import time

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def record_rtsp_stream(rtsp_url, output_video_path, output_csv_path):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Open CSV file for writing timestamps
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Timestamp'])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Write the frame to the output video
            out.write(frame)

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Write frame number and timestamp to CSV
            csv_writer.writerow([frame_count, timestamp])

            frame_count += 1

            # Display the frame (optional, for debugging)
            cv2.imshow('RTSP Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
rtsp_url = "rtsp://wowza01.bellevuewa.gov:1935/live/CCTV047NE.stream"
output_video_path = f"output_video_{current_time}.mp4"
output_csv_path = f"frame_timestamps_{current_time}.csv"

record_rtsp_stream(rtsp_url, output_video_path, output_csv_path)