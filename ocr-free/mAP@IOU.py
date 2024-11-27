import json
import os
from PIL import Image

# Convert pixel coordinates to normalized coordinates
def pixel_to_normalized(pixel_box, image_width, image_height):
    """
    Convert pixel coordinates to normalized coordinates (0 to 1 scale).

    Args:
        pixel_box (tuple): Bounding box in pixel format (x1, y1, x2, y2).
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Normalized bounding box coordinates.
    """
    x1, y1, x2, y2 = pixel_box
    return (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height)

# Merge multiple bounding boxes into a single box
def merge_boxes(boxes):
    """
    Merge multiple bounding boxes into one encompassing bounding box.

    Args:
        boxes (list): List of bounding boxes.

    Returns:
        tuple: Bounding box that encompasses all input boxes.
    """
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return (x1, y1, x2, y2)

# Calculate Intersection over Union (IoU) for two bounding boxes
def calculate_iou(predicted_box, ground_truth_box):
    """
    Calculate IoU between two bounding boxes.

    Args:
        predicted_box (tuple): Predicted bounding box.
        ground_truth_box (tuple): Ground truth bounding box.

    Returns:
        tuple: IoU, intersection area, predicted area, and ground truth area.
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_box

    # Calculate intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou, inter_area, pred_area, gt_area

# Calculate Dice coefficient for two bounding boxes
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

# Calculate mean Average Precision (mAP) at multiple IoU thresholds
def calculate_mAP(results, iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
    """
    Calculate mean Average Precision (mAP) at multiple IoU thresholds.

    Args:
        results (list): List of evaluation results containing IoU values.
        iou_thresholds (list): List of IoU thresholds for evaluation.

    Returns:
        float: Mean Average Precision score.
    """
    precisions = []
    
    for threshold in iou_thresholds:
        true_positives = sum(1 for result in results if result['iou'] >= threshold)
        total_predictions = len(results)

        # Calculate precision
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0
        precisions.append(precision)

    # Calculate mean Average Precision
    mAP = sum(precisions) / len(iou_thresholds) if precisions else 0.0
    return mAP

# Evaluate predictions by comparing ground truth and predicted bounding boxes
def evaluate_predictions(ground_truth_file, predicted_folder, image_folder):
    """
    Evaluate predictions by calculating IoU and Dice coefficients.

    Args:
        ground_truth_file (str): Path to the ground truth JSON file.
        predicted_folder (str): Folder containing predicted JSON files.
        image_folder (str): Folder containing corresponding images.

    Returns:
        list: Evaluation results including IoU and Dice coefficients.
    """
    # Load ground truth data
    with open(ground_truth_file, 'r') as gt_file:
        ground_truth_data = json.load(gt_file)
        print("Loaded ground truth data successfully.")

    results = []

    # Iterate through predicted files
    for pred_filename in os.listdir(predicted_folder):
        if not pred_filename.endswith(".json"):
            continue

        # Construct image path from predicted filename
        common_image_name = pred_filename.split("_output")[0].replace("10.14-", "")
        image_path = os.path.join(image_folder, f"{common_image_name}.png")
        print(f"Constructed image path: {image_path}")

        predicted_file = os.path.join(predicted_folder, pred_filename)

        # Load the image and get its dimensions
        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            continue

        # Find matching ground truth entry
        matching_gt_entries = [entry for entry in ground_truth_data if common_image_name in entry["image_path"]]
        if not matching_gt_entries:
            print(f"No matching ground truth entry found for: {common_image_name}")
            continue

        # Load the predicted JSON file
        with open(predicted_file, 'r') as pred_file:
            predicted_data = json.load(pred_file)
            print(f"Loaded predicted data for {common_image_name}.")

        parsed_predictions = None
        try:
            for item in predicted_data[0]["generated_text"]:
                if item["role"] == "assistant":
                    parsed_predictions = json.loads(item["content"])
                    break
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error extracting and parsing JSON content from predictions: {e}")
            continue

        if parsed_predictions is None:
            print(f"No valid parsed predictions found for: {common_image_name}")
            continue

        # Compare predictions with ground truth
        for gt_entry, pred_entry in zip(matching_gt_entries, parsed_predictions):
            print(f"Processing entry for image: {common_image_name}.")

            try:
                gt_box_data = gt_entry["conversations"][1]["box_data"]["box"]
                ground_truth_box = pixel_to_normalized(gt_box_data, image_width, image_height)
            except (KeyError, IndexError) as e:
                print(f"Ground truth box data not found in entry for {common_image_name}: {e}")
                continue

            if "details" not in pred_entry:
                print(f"'details' key not found in predicted entry for question: {pred_entry.get('question', 'Unknown')}")
                continue

            # Process bounding boxes
            if isinstance(pred_entry["details"], list):
                predicted_boxes = []
                for detail in pred_entry["details"]:
                    if "bounding_box" not in detail:
                        print(f"'bounding_box' key not found in detail for question: {pred_entry.get('question', 'Unknown')}")
                        continue

                    box_str = detail["bounding_box"].replace("[", "(").replace("]", ")").strip("()").split("), (")
                    try:
                        box_coords = [tuple(map(float, coord.split(","))) for coord in box_str]
                        predicted_boxes.append((box_coords[0][0], box_coords[0][1], box_coords[1][0], box_coords[1][1]))
                    except ValueError as e:
                        print(f"Error converting bounding box to coordinates for question: {pred_entry.get('question', 'Unknown')} - {e}")
                        continue

                merged_predicted_box = merge_boxes(predicted_boxes) if predicted_boxes else None
            else:
                if "bounding_box" not in pred_entry["details"]:
                    continue

                box_str = pred_entry["details"]["bounding_box"].replace("[", "(").replace("]", ")").strip("()").split("), (")
                try:
                    box_coords = [tuple(map(float, coord.split(","))) for coord in box_str]
                    merged_predicted_box = (box_coords[0][0], box_coords[0][1], box_coords[1][0], box_coords[1][1])
                except ValueError as e:
                    continue

            # Calculate IoU and Dice coefficients
            iou, inter_area, pred_area, gt_area = calculate_iou(merged_predicted_box, ground_truth_box)
            dice = calculate_dice(inter_area, pred_area, gt_area)

            # Append results
            results.append({
                "question": pred_entry["question"],
                "iou": iou,
                "dice": dice,
                "ground_truth_box": ground_truth_box,
                "predicted_box": merged_predicted_box
            })

    # Calculate and display mAP
    mAP = calculate_mAP(results)
    print(f"mAP @ IoU [0.50:0.95]: {mAP:.4f}")
    return results

if __name__ == "__main__":
    # Specify dataset parameters
    dataset_name = "CORD"  # Change to "FUNSD" or "SROIE" for other datasets
    ground_truth_file = f'datasets/updated-VIE_{dataset_name}.json'
    predicted_folder = f'ocr-free/results/3-information extraction/{dataset_name}/master'
    image_folder = 'datasets/images'

    # Run evaluation
    evaluation_results = evaluate_predictions(ground_truth_file, predicted_folder, image_folder)

    # Display evaluation results
    if evaluation_results:
        for result in evaluation_results:
            print(f"Question: {result['question']}, IoU: {result['iou']}, Dice Coefficient: {result['dice']}")
    else:
        print("No results were generated. Please check the input data and code logic.")
