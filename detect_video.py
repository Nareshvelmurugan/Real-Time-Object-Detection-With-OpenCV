# Run MobileNetSSD object detection on a video file and save the annotated result.
# Unlike real_time_object_detection.py, this needs no webcam or display.
#
# How to run:
# python detect_video.py --prototxt MobileNetSSD_deploy.prototxt.txt \
#     --model MobileNetSSD_deploy.caffemodel --input in.mp4 --output out.mp4

import argparse
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", required=True,
	help="path to the input video file")
ap.add_argument("-o", "--output", required=True,
	help="path of the annotated video to write (.mp4)")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak predictions")
ap.add_argument("--start", type=float, default=0.0,
	help="second of the input to start at")
ap.add_argument("--duration", type=float, default=None,
	help="seconds of the input to process (default: to the end)")
ap.add_argument("--title", default=None,
	help="optional caption drawn in the top-left corner")
args = vars(ap.parse_args())

# Class labels in the order MobileNetSSD was trained with (index 0 is background)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Fixed seed so every class keeps the same colour between runs
COLORS = np.random.default_rng(7).uniform(60, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture(args["input"])
if not cap.isOpened():
	raise SystemExit("[ERROR] could not open input video: " + args["input"])

src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(args["start"] * src_fps))
max_frames = None if args["duration"] is None else int(args["duration"] * src_fps)

writer = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

print("[INFO] processing video...")
frames = 0
started = time.time()
while max_frames is None or frames < max_frames:
	ok, frame = cap.read()
	if not ok:
		break

	# Same pre-processing as the real-time script: 300x300, scaled to [-1, 1]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), (1/127.5), (300, 300), 127.5, swapRB=True)
	net.setInput(blob)
	predictions = net.forward()

	counts = {}
	for i in np.arange(0, predictions.shape[2]):
		confidence = predictions[0, 0, i, 2]
		if confidence < args["confidence"]:
			continue
		idx = int(predictions[0, 0, i, 1])
		box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		color = COLORS[idx]
		counts[CLASSES[idx]] = counts.get(CLASSES[idx], 0) + 1

		label = "{}: {:.0f}%".format(CLASSES[idx], confidence * 100)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# Filled tag behind the label so it stays readable on any background
		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		y = startY if startY - th - 8 > 0 else startY + th + 8
		cv2.rectangle(frame, (startX, y - th - 8), (startX + tw + 8, y), color, -1)
		cv2.putText(frame, label, (startX + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

	# Status bar: optional title plus a per-class count for this frame
	summary = ", ".join("{} {}".format(n, c) for c, n in sorted(counts.items())) or "no detections"
	text = summary if args["title"] is None else "{}  |  {}".format(args["title"], summary)
	cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
	cv2.putText(frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

	writer.write(frame)
	frames += 1

cap.release()
writer.release()
elapsed = time.time() - started
print("[INFO] Frames written: {}".format(frames))
print("[INFO] Elapsed Time: {:.2f}".format(elapsed))
print("[INFO] Approximate FPS: {:.2f}".format(frames / elapsed if elapsed else 0))
