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
    """
    Calculates the euclidian distance between two points of a bounding boxes.

    Args:
        boxA (list or tuple): The coordinates of the first box (left, right, top, bottom).
        boxB (list or tuple): The coordinates of the second box (left, right, top, bottom).

    Returns:
        float: distance between boxA and boxB left top corner.
    """
    if None in (boxA, boxB):
        return math.inf
    sum_sq_diff = (boxA[0] - boxB[0])**2 + (boxA[2] - boxB[2])**2
    return math.sqrt(sum_sq_diff)


def crop_with_padding(image, box, multiplier):
    """
    Crops an image based on a bounding box, adding padding equal to the
    size of the original crop, effectively doubling its dimensions.

    Args:
        image (np.array): The input image loaded by OpenCV.
        box (tuple): A tuple containing the bounding box coordinates
                     in the format (left, right, top, bottom).
        padding multiplier(integer): box multiplier used for padding

    Returns:
        np.array: The cropped image with padding.
    """
    # Get the dimensions of the full image
    img_h, img_w, _ = image.shape

    # Unpack the bounding box coordinates
    left, right, top, bottom = box

    # Calculate the width and height of the original crop
    width = right - left
    height = bottom - top

    # Calculate the padding to add to each side (half of the original dimensions)
    pad_x = width * multiplier
    pad_y = height * multiplier

    # Calculate the new coordinates for the padded crop
    new_left = left - pad_x
    new_right = right + pad_x
    new_top = top - pad_y
    new_bottom = bottom + pad_y

    # --- IMPORTANT: Clamp coordinates to the image boundaries ---
    # This prevents errors if the padded box goes outside the image
    final_left = max(0, new_left)
    final_right = min(img_w, new_right)
    final_top = max(0, new_top)
    final_bottom = min(img_h, new_bottom)

    # Crop the image using the final, clamped coordinates
    padded_crop = image[final_top:final_bottom, final_left:final_right]

    return padded_crop