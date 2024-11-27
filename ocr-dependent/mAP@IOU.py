import json
import os
from PIL import Image
import re
import numpy as np

# Calculate precision and recall at multiple IoU thresholds
def calculate_precision_recall_at_thresholds(results, iou_thresholds=np.arange(0.50, 1.00, 0.05)):
    """
    Calculate precision and recall at multiple IoU thresholds.

    Args:
        results (list): List of dictionaries containing IoU values for predictions.
        iou_thresholds (numpy array): IoU thresholds for evaluation.

    Returns:
        tuple: Lists of precision and recall values for each threshold.
    """
    iou_thresholds = sorted(iou_thresholds)
    precisions, recalls = [], []

    for threshold in iou_thresholds:
        # Count true positives and false positives
        tp = sum(1 for result in results if result['iou'] >= threshold)
        fp = sum(1 for result in results if result['iou'] < threshold)
        fn = 0  # False negatives are not explicitly considered in this implementation

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

# Calculate average precision given precision and recall values
def calculate_average_precision(precisions, recalls):
    """
    Calculate the average precision given lists of precision and recall values.

    Args:
        precisions (list): Precision values for different thresholds.
        recalls (list): Recall values for different thresholds.

    Returns:
        float: Average precision score.
    """
    precisions = [0] + precisions + [0]  # Add boundary values for precision
    recalls = [0] + recalls + [1]  # Add boundary values for recall

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Calculate area under the precision-recall curve
    indices = [i for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1]]
    average_precision = sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)
    return average_precision

# Calculate mean Average Precision (mAP) at multiple IoU thresholds
def calculate_map_at_iou_thresholds(results, iou_thresholds=np.arange(0.50, 1.00, 0.05)):
    """
    Calculate the mean Average Precision (mAP) at multiple IoU thresholds.

    Args:
        results (list): List of evaluation results with IoU values.
        iou_thresholds (numpy array): IoU thresholds for evaluation.

    Returns:
        float: Mean Average Precision score.
    """
    precisions, recalls = calculate_precision_recall_at_thresholds(results, iou_thresholds)
    average_precisions = [calculate_average_precision([precisions[i]], [recalls[i]]) for i in range(len(iou_thresholds))]
    map_score = np.mean(average_precisions)
    return map_score

# Convert bounding box from pixel to normalized coordinates
def pixel_to_normalized(pixel_box, image_width, image_height):
    """
    Convert pixel coordinates to normalized coordinates.

    Args:
        pixel_box (tuple): Bounding box in pixel format (x1, y1, x2, y2).
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Normalized bounding box.
    """
    x1, y1, x2, y2 = pixel_box
    return (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height)

# Merge multiple bounding boxes into a single box
def merge_boxes(boxes):
    """
    Merge multiple bounding boxes into one encompassing box.

    Args:
        boxes (list): List of bounding boxes.

    Returns:
        tuple: Merged bounding box.
    """
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return (x1, y1, x2, y2)

# Calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(predicted_box, ground_truth_box):
    """
    Calculate IoU between two bounding boxes.

    Args:
        predicted_box (tuple): Predicted bounding box.
        ground_truth_box (tuple): Ground truth bounding box.

    Returns:
        tuple: IoU, intersection area, predicted box area, and ground truth area.
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_box

    # Calculate intersection area
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou, inter_area, pred_area, gt_area

# Calculate Dice coefficient between two bounding boxes
def calculate_dice(inter_area, pred_area, gt_area):
    """
    Calculate Dice coefficient between two bounding boxes.

    Args:
        inter_area (float): Intersection area between the boxes.
        pred_area (float): Area of the predicted box.
        gt_area (float): Area of the ground truth box.

    Returns:
        float: Dice coefficient.
    """
    dice = (2 * inter_area) / (pred_area + gt_area) if (pred_area + gt_area) != 0 else 0
    return dice

# Parse bounding box from various formats into a standard format
def parse_bounding_box(bbox):
    """
    Parse bounding box from different formats into a standard format.

    Args:
        bbox (various): Bounding box in various formats.

    Returns:
        tuple or None: Standardized bounding box or None if format is invalid.
    """
    if isinstance(bbox, list):
        if len(bbox) == 2 and isinstance(bbox[0], list) and isinstance(bbox[1], list):
            return bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        elif len(bbox) == 4 and all(isinstance(coord, (float, int)) for coord in bbox):
            return bbox[0], bbox[1], bbox[2], bbox[3]
    elif isinstance(bbox, str):
        coords = re.findall(r"[\d\.]+", bbox)
        if len(coords) == 4:
            return float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
    return None

# Clean JSON content by removing formatting markers and decode it
def clean_json_format(file_path):
    """
    Clean and parse JSON content from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON content.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace("```json", "").replace("```", "").strip()
    unescaped_content = json.loads(content)
    cleaned_content = json.loads(unescaped_content)
    return cleaned_content

# Evaluate predictions by comparing ground truth and predicted bounding boxes
def evaluate_predictions(ground_truth_file, predicted_folder, image_folder):
    """
    Evaluate predictions by calculating IoU and Dice coefficients.

    Args:
        ground_truth_file (str): Path to ground truth JSON file.
        predicted_folder (str): Folder containing predicted JSON files.
        image_folder (str): Folder containing corresponding images.

    Returns:
        list: Evaluation results including IoU and Dice coefficients.
    """
    with open(ground_truth_file, 'r') as gt_file:
        ground_truth_data = json.load(gt_file)

    results = []
    for pred_filename in os.listdir(predicted_folder):
        if not pred_filename.endswith(".json"):
            continue

        common_image_name = pred_filename.split("_output8b")[0].replace("pixtralPinakiprompt-", "")
        image_path = os.path.join(image_folder, f"{common_image_name}.png")
        predicted_file = os.path.join(predicted_folder, pred_filename)

        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except FileNotFoundError:
            continue

        matching_gt_entries = [entry for entry in ground_truth_data if common_image_name in entry["image_path"]]
        if not matching_gt_entries:
            continue

        with open(predicted_file, 'r') as pred_file:
            predicted_data = clean_json_format(pred_file)

        for gt_entry, pred_entry in zip(matching_gt_entries, predicted_data):
            if not isinstance(pred_entry, dict):
                continue
            gt_box_data = gt_entry["conversations"][1]["box_data"]["box"]
            ground_truth_box = pixel_to_normalized(gt_box_data, image_width, image_height)

            predicted_boxes = [parse_bounding_box(detail["bounding_box"]) for detail in pred_entry["details"]]
            merged_predicted_box = merge_boxes(predicted_boxes)

            iou, inter_area, pred_area, gt_area = calculate_iou(merged_predicted_box, ground_truth_box)
            dice = calculate_dice(inter_area, pred_area, gt_area)

            results.append({
                "question": pred_entry["question"],
                "iou": iou,
                "dice": dice,
                "ground_truth_box": ground_truth_box,
                "predicted_box": merged_predicted_box
            })

    return results

# Paths to required data files
ground_truth_file = 'datasets/updated-VIE_CORD.json'
predicted_folder = 'ocr-dependent/results/3-information extraction/cord'
image_folder = 'datasets/images/CORD'

# Evaluate predictions and calculate mAP
evaluation_results = evaluate_predictions(ground_truth_file, predicted_folder, image_folder)
iou_thresholds = np.arange(0.50, 1.00, 0.05)
map_score = calculate_map_at_iou_thresholds(evaluation_results, iou_thresholds)

# Display results
print(f"mAP @ IoU [0.50:0.95]: {map_score:.4f}")
if evaluation_results:
    for result in evaluation_results:
        print(f"Question: {result['question']}, IoU: {result['iou']}, Dice Coefficient: {result['dice']}")
