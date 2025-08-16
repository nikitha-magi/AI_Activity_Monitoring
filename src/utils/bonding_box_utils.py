"Utility function to process and handle bonding boxes"
import math

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list or tuple): The coordinates of the first box (left, right, top, bottom).
        boxB (list or tuple): The coordinates of the second box (left, right, top, bottom).

    Returns:
        float: The IoU value, between 0 and 1.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    # box format: (left, right, top, bottom)
    xA = max(boxA[0], boxB[0]) # Max of the two left values
    yA = max(boxA[2], boxB[2]) # Max of the two top values
    xB = min(boxA[1], boxB[1]) # Min of the two right values
    yB = min(boxA[3], boxB[3]) # Min of the two bottom values

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2]) # (right - left) * (bottom - top)
    boxBArea = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2]) # (right - left) * (bottom - top)

    # Compute the intersection over union, handling the case of zero area
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    iou = interArea / denominator

    return iou


def calculate_distance(boxA, boxB):
    if None in (boxA, boxB):
        return math.inf
    sum_sq_diff = (boxA[0] - boxB[0])**2 + (boxA[2] - boxB[2])**2
    return math.sqrt(sum_sq_diff)