# Utilities for object detector.

import numpy as np
import tensorflow as tf
from threading import Thread
import cv2
from src.utils import label_map_util
from src.utils.bonding_box_utils import calculate_iou
from src.utils.config import MODEL_DIR

detection_graph = tf.Graph()

# score threshold for showing bounding boxes.
_score_thresh = 0.27

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_DIR / 'frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = MODEL_DIR / 'hand_label_map.pbtxt'

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("====== loading HAND frozen graph into memory ======")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print("+++++++++++++ Hand Inference graph loaded +++++++++++++")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    # p1 = (int(left), int(top))
    # p2 = (int(right), int(bottom))
    # cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
    predictions = []
    for index, score in enumerate(scores):
        if score < score_thresh:
            break
        predictions.append((score, boxes[index][1] * im_width, boxes[index][3] * im_width,
                            boxes[index][0] * im_height, boxes[index][2] * im_height))
    
    filtered_predictions = non_max_suppression(predictions, iou_threshold=0.01)
    hand_detected = []
    for filtered_prediction in filtered_predictions[:num_hands_detect]:
        (left, right, top, bottom) = (int(filtered_prediction[1]), int(filtered_prediction[2]),
                                int(filtered_prediction[3]), int(filtered_prediction[4]))
        hand_detected.append((left, right, top, bottom))
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
    return hand_detected

def non_max_suppression(predictions, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes.

    Args:
        predictions (list): A list of prediction tuples, where each tuple is
                            (score, left, right, top, bottom).
        iou_threshold (float): The IoU threshold to use for filtering. Boxes with
                               IoU >= this threshold will be suppressed.

    Returns:
        list: A list of the filtered predictions.
    """
    if not predictions:
        return []

    # Sort predictions by their confidence score in descending order
    predictions.sort(key=lambda x: x[0], reverse=True)

    final_predictions = []
    
    while predictions:
        # Take the prediction with the highest score
        current_pred = predictions.pop(0)
        final_predictions.append(current_pred)
        
        # Get the bounding box of the current prediction
        current_box = current_pred[1:]

        # Create a new list to hold predictions that don't overlap
        preds_to_keep = []
        for pred in predictions:
            box_to_compare = pred[1:]
            
            # Calculate IoU
            iou = calculate_iou(current_box, box_to_compare)
            
            # If the IoU is below the threshold, keep the prediction
            if iou < iou_threshold:
                preds_to_keep.append(pred)
        
        # Update the list for the next iteration
        predictions = preds_to_keep

    return final_predictions

# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
