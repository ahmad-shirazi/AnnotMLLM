from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import json
import cv2
import numpy as np
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from transformers import pipeline as transformers_pipeline
from vllm import LLM
from vllm.sampling_params import SamplingParams
import base64
import base64
import ast
import json
import re

class OCR:
    def __init__(self):
        """
        Initializes the OCR class with a text detection and recognition predictor.
        """
        self.predictor = ocr_predictor(det_arch='db_resnet50', pretrained=True, assume_straight_pages=True)

    def process(self, file_path):
        """
        Processes an image file and performs OCR.

        Args:
            file_path (str): Path to the image file.

        Returns:
            dict: OCR results from the predictor.
        """
        self.doc = DocumentFile.from_images(file_path)
        self.result = self.predictor(self.doc)
        return self.result

    def get_bounding_boxes(self):
        """
        Extracts bounding boxes from the OCR results and assigns unique IDs.

        Returns:
            list: List of bounding boxes with unique IDs.
        """
        self.OCRout = self.result.export()
        boxes_with_ids = []
        box_id = 0
        for block in self.OCRout['pages'][0]['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    boxes_with_ids.append({
                        'id': f'BB{box_id}',
                        'bounding_box': word['geometry']
                    })
                    box_id += 1
        return boxes_with_ids

    def generate_cropped_images_with_ids(self, output_image_path):
        """
        Creates a single image with cropped bounding box regions and IDs.

        Args:
            output_image_path (str): Path to save the output image.
        """
        boxes_with_ids = self.get_bounding_boxes()
        top_margin = 20
        total_height = 0

        for box in boxes_with_ids:
            y1 = int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            y2 = int(box['bounding_box'][1][1] * self.doc[0].shape[0])
            box_height = max(0, y2 - y1)
            total_height += box_height + 10

        total_height += top_margin
        img_width = 1000
        final_image = np.ones((total_height, img_width, 3), dtype=np.uint8) * 255
        y_offset = top_margin

        for i, box in enumerate(boxes_with_ids):
            x1, y1 = int(box['bounding_box'][0][0] * self.doc[0].shape[1]), int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            x2, y2 = int(box['bounding_box'][1][0] * self.doc[0].shape[1]), int(box['bounding_box'][1][1] * self.doc[0].shape[0])

            if y2 <= y1 or x2 <= x1:
                print(f"Skipping invalid bounding box for BB{i}: height or width is zero or negative.")
                continue

            cropped_image = self.doc[0][y1:y2, x1:x2]
            crop_height, crop_width, _ = cropped_image.shape

            if crop_height <= 0 or crop_width <= 0:
                print(f"Skipping BB{i}: Cropped image has zero or negative dimensions.")
                continue

            if y_offset + crop_height > final_image.shape[0]:
                print(f"Error: Insufficient space in final image to place cropped image for BB{i}.")
                break

            final_image[y_offset:y_offset + crop_height, 0:crop_width] = cropped_image
            font_scale = crop_height / 50
            font_thickness = max(1, int(font_scale * 2))
            text_color = (0, 0, 0)

            text = f"{box['id']}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = crop_width + 10
            text_y = y_offset + (crop_height // 2) + (text_size[1] // 2)

            cv2.putText(final_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            y_offset += crop_height + 10

        cv2.imwrite(output_image_path, final_image)


class VLM:
    def __init__(self, token):
        """
        Initializes the VLM class by logging in with a token.
        """
        login(token)

    def model(self, model_id, out_json):
        """
        Loads the specified vision-language model and its processor.

        Args:
            model_id (str): Model identifier.
            out_json (bool): Whether to return JSON output.
        """
        self.output_json = out_json
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process(self, image_path, input_text):
        """
        Processes an image and input text to generate output from the model.

        Args:
            image_path (str): Path to the input image.
            input_text (str): Text instruction for the model.

        Returns:
            str: Model output as decoded text or JSON.
        """
        image = Image.open(image_path)
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        text_inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)
        inputs = {'input_ids': text_inputs['input_ids'], 'pixel_values': image_inputs['pixel_values']}

        output = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)

        if self.output_json:
            try:
                return json.dumps(json.loads(decoded_output), indent=4)
            except json.JSONDecodeError:
                print("The model output is not valid JSON:", decoded_output)
        return decoded_output


class PixtralProcessor:
    def __init__(self, model_name="mistralai/Pixtral-12B-2409", max_tokens=150192):
        """
        Initializes the PixtralProcessor class with a specific model.

        Args:
            model_name (str): Identifier for the Pixtral model.
            max_tokens (int): Maximum number of tokens for generation.
        """
        self.model_name = model_name
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        self.llm = LLM(model=self.model_name, tokenizer_mode="mistral", limit_mm_per_prompt={"image": 5}, max_model_len=32768)

    def image_to_data_uri(self, image_path):
        """
        Converts a local image file to a data URI.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Data URI representation of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

    def pixtral(self, image_path1, image_path2, prompt):
        """
        Processes two images and a prompt using the Pixtral model.

        Args:
            image_path1 (str): Path to the first image.
            image_path2 (str): Path to the second image.
            prompt (str): Prompt text for processing.

        Returns:
            str: Model output.
        """
        image_data_uri1 = self.image_to_data_uri(image_path1)
        image_data_uri2 = self.image_to_data_uri(image_path2)
        messages = [
            {"role": "system", "content": "You identify bounding boxes based on given inputs."},
            {"role": "user", "content": [{"type": "text", "text": prompt},
                                         {"type": "image_url", "image_url": {"url": image_data_uri1}},
                                         {"type": "image_url", "image_url": {"url": image_data_uri2}}]}
        ]
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

class Annotation:
    def __init__(self):
        """
        Initializes the Annotation class.
        """
        pass

    def flatten_bounding_box(self, bounding_box):
        """
        Flattens a bounding box if it is nested within a single-element list.

        Args:
            bounding_box (list): The bounding box, potentially nested.

        Returns:
            list: The flattened bounding box or the original if already flat.
        """
        if len(bounding_box) == 1 and isinstance(bounding_box[0], list):
            return bounding_box[0]
        return bounding_box

    def draw_bounding_boxes(self, image_path, json_data, output_path):
        """
        Draws bounding boxes on an image based on the provided JSON data.

        Args:
            image_path (str): Path to the input image.
            json_data (dict): JSON data containing bounding box information.
            output_path (str): Path to save the output image with bounding boxes.
        """
        # Load the image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Iterate through the JSON data
        for key, value in json_data.items():
            if isinstance(value, list):  # Handle multiple bounding boxes
                for box in value:
                    bounding_box = self.flatten_bounding_box(box["bounding_box"])
                    x1 = int(bounding_box[0][0] * image_width)
                    y1 = int(bounding_box[0][1] * image_height)
                    x2 = int(bounding_box[1][0] * image_width)
                    y2 = int(bounding_box[1][1] * image_height)

                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif "bounding_box" in value and value["bounding_box"] is not None:
                bounding_box = self.flatten_bounding_box(value["bounding_box"])
                x1 = int(bounding_box[0][0] * image_width)
                y1 = int(bounding_box[0][1] * image_height)
                x2 = int(bounding_box[1][0] * image_width)
                y2 = int(bounding_box[1][1] * image_height)

                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the annotated image
        cv2.imwrite(output_path, image)

    def annotate_image_with_text_recognition(self, image_path, ocr_result, output_path):
        """
        Annotates an image with bounding boxes and corresponding recognized text.

        Args:
            image_path (str): Path to the input image.
            ocr_result (list): OCR result containing text and bounding box information.
            output_path (str): Path to save the annotated image.
        """
        # Load the image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Iterate through OCR results
        for word_data in ocr_result:
            word = word_data['value']
            bounding_box = word_data['bounding_box']
            x1 = int(bounding_box[0][0] * image_width)
            y1 = int(bounding_box[0][1] * image_height)
            x2 = int(bounding_box[1][0] * image_width)
            y2 = int(bounding_box[1][1] * image_height)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add the recognized text near the bounding box
            cv2.putText(image, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the annotated image
        cv2.imwrite(output_path, image)


# Function to extract the code block within triple backticks
def extract_code_block(s):
    """
    Extracts the code block enclosed in triple backticks from a string.

    Args:
        s (str): Input string containing a code block.

    Returns:
        str: Extracted code block.

    Raises:
        ValueError: If no code block is found in the input string.
    """
    pattern = r'```(?:[\w]+\n)?(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        raise ValueError('No code block found in the string.')
    return match.group(1).strip()

# Function to clean the code block (remove "information =")
def clean_code_block(code_block):
    """
    Removes specific prefixes (like "information =") from the code block.

    Args:
        code_block (str): Input code block.

    Returns:
        str: Cleaned code block.
    """
    return code_block.replace('information =', '').strip()

# Function to convert tuples to lists and handle floating-point number pairs
def convert_tuples_to_lists(s):
    """
    Converts tuples to lists in a string and ensures bounding boxes are formatted correctly.

    Args:
        s (str): Input string.

    Returns:
        str: Converted string with tuples replaced by lists.
    """
    s = s.replace('(', '[').replace(')', ']')
    s = re.sub(r'\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]', 
               r'[[\1, \2], [\3, \4]]', s)
    return s

# Function to correct malformed floats
def sanitize_malformed_floats(s):
    """
    Fixes malformed floating-point numbers in a string.

    Args:
        s (str): Input string.

    Returns:
        str: Corrected string.
    """
    return re.sub(r'(\d+\.\d+)\.\d+', r'\1', s)

# Improved function to balance brackets
def balance_brackets(s):
    """
    Balances mismatched brackets in a string.

    Args:
        s (str): Input string.

    Returns:
        str: String with balanced brackets.

    Raises:
        ValueError: If mismatched brackets cannot be resolved.
    """
    stack = []
    balanced_s = []

    for char in s:
        if char in "{[":
            stack.append(char)
        elif char in "]}":
            if stack:
                last_open = stack.pop()
                if (last_open == '{' and char != '}') or (last_open == '[' and char != ']'):
                    raise ValueError("Mismatched brackets found")
            else:
                char = '}' if char == ']' else ']'
        balanced_s.append(char)

    while stack:
        last_open = stack.pop()
        balanced_s.append('}' if last_open == '{' else ']')

    return ''.join(balanced_s)

# Function to parse the string into a dictionary
def parse_to_dict(s):
    """
    Parses a string into a Python dictionary.

    Args:
        s (str): Input string.

    Returns:
        dict: Parsed dictionary.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        s = balance_brackets(s)
        return ast.literal_eval(s)
    except SyntaxError as e:
        raise ValueError(f"Error parsing string to dict: {e}")

# Function to make sure the parsed dictionary is JSON serializable
def make_json_serializable(data):
    """
    Ensures a Python object is JSON serializable.

    Args:
        data (dict): Input data.

    Returns:
        dict: JSON serializable data.
    """
    return json.loads(json.dumps(data))

# Helper function to remove quotes around coordinates inside bounding boxes
def remove_quotes_in_bounding_box(data):
    """
    Removes quotes around coordinates inside bounding boxes in a dictionary.

    Args:
        data (dict or list): Input data.

    Returns:
        dict or list: Data with cleaned bounding boxes.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "bounding_box":
                data[key] = [
                    [[float(coord) for coord in box_pair.strip("[]").split(", ")] if isinstance(box_pair, str) else box_pair
                     for box_pair in box]
                    if isinstance(box, list) else box
                    for box in value
                ]
            elif isinstance(value, (dict, list)):
                remove_quotes_in_bounding_box(value)
    elif isinstance(data, list):
        for item in data:
            remove_quotes_in_bounding_box(item)
    return data

# Function to extract and clean the JSON from a string
def extract_json_from_string(s):
    """
    Extracts and cleans JSON from a string.

    Args:
        s (str): Input string.

    Returns:
        dict: Extracted and cleaned JSON data.
    """
    code_block = extract_code_block(s)
    code_block = clean_code_block(code_block)
    code_block = convert_tuples_to_lists(code_block)
    code_block = sanitize_malformed_floats(code_block)
    code_block = balance_brackets(code_block)
    data = parse_to_dict(code_block)
    return make_json_serializable(remove_quotes_in_bounding_box(data))

# Function to merge bounding boxes into a single bounding box
def merge_bounding_boxes(bounding_boxes):
    """
    Merges multiple bounding boxes into a single bounding box.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Single merged bounding box.
    """
    if isinstance(bounding_boxes[0][0], float) and isinstance(bounding_boxes[0][1], float):
        return bounding_boxes

    x1, y1 = bounding_boxes[0][0]
    second_corners = [box[1] for box in bounding_boxes]
    max_x = max([corner[0] for corner in second_corners])
    max_y = max([corner[1] for corner in second_corners])

    return [[x1, y1], [max_x, max_y]]

# Function to correct bounding boxes
def correct_bounding_boxes(bounding_boxes):
    """
    Corrects and standardizes bounding boxes.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Corrected bounding boxes.
    """
    corrected_boxes = []

    def flatten_box(box):
        if isinstance(box, list) and len(box) == 1 and isinstance(box[0], list):
            return flatten_box(box[0])
        return box

    all_numbers = []
    for box in bounding_boxes:
        flat_box = flatten_box(box)
        if isinstance(flat_box, list) and all(isinstance(coord, float) for coord in flat_box):
            all_numbers.extend(flat_box)
        else:
            for pair in flat_box:
                all_numbers.extend(pair)

    if len(all_numbers) % 4 != 0:
        print(f"Skipping invalid bounding box format with {len(all_numbers)} numbers: {bounding_boxes}")
        return corrected_boxes

    if len(all_numbers) == 4:
        return [[all_numbers[0], all_numbers[1]], [all_numbers[2], all_numbers[3]]]

    for i in range(0, len(all_numbers), 4):
        corrected_boxes.append([[all_numbers[i], all_numbers[i+1]], [all_numbers[i+2], all_numbers[i+3]]])

    return corrected_boxes
