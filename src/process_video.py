import cv2

import numpy as np
import pandas as pd
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

from view_transformer import ViewTransformer


# CONFIGURATION
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 1280


def process_video(source_video_path: str, target_video_path: str, model_name: str, output_csv: str) -> None:
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 250
    SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
    TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1],])

    # TRANSFORM PERSPECTIVE
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    model = YOLO(model_name)

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    # tracker initiation
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # annotators configuration
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    output_data = []

    # open target video
    with sv.VideoSink(target_video_path, video_info) as sink:

        # loop over source video frame
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # filter out detections by class and confidence
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            detections = detections[detections.class_id == 2]

            # filter out detections outside the zone
            detections = detections[polygon_zone.trigger(detections)]

            # refine detections using non-max suppression
            detections = detections.with_nms(IOU_THRESHOLD)

            # pass detection through the tracker
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )

            # calculate the detections position inside the target RoI
            points = view_transformer.transform_points(points=points).astype(int)

            # store detections position
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            # format labels
            labels = []

            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                    speed = -1
                else:
                    # calculate speed
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                output_data.append([tracker_id, len(coordinates[tracker_id]), *detections[detections.tracker_id == tracker_id].xyxy[0].tolist(), speed])

            # annotate frame
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # add frame to target video
            sink.write_frame(annotated_frame)

    df = pd.DataFrame(data=output_data, columns=["tracker_id", "frames", "x1", "y1", "x2", "y2", "speed"])
    df.to_csv(output_csv, index=False)
    print("Output saved to ", output_csv)
