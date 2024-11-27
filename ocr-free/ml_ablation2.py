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
import ast
import json
import re

class OCR:
    def __init__(self):
        """
        Initialize the OCR class with a pre-trained text detection model.
        """
        self.predictor = ocr_predictor(det_arch='db_resnet50', pretrained=True, assume_straight_pages=True)

    def process(self, file_path):
        """
        Process the input file to extract text and bounding boxes.

        Args:
            file_path (str): Path to the input image or document.

        Returns:
            dict: OCR results.
        """
        self.doc = DocumentFile.from_images(file_path)
        self.result = self.predictor(self.doc)
        return self.result

    def get_bounding_boxes(self):
        """
        Extract bounding boxes and assign unique IDs.

        Returns:
            list: A list of dictionaries containing bounding box IDs and coordinates.
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
        Generate an annotated image showing cropped bounding boxes with unique IDs.

        Args:
            output_image_path (str): Path to save the annotated image.
        """
        boxes_with_ids = self.get_bounding_boxes()
        top_margin = 20
        total_height = 0

        # Calculate total height for final image
        for box in boxes_with_ids:
            y1 = int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            y2 = int(box['bounding_box'][1][1] * self.doc[0].shape[0])
            box_height = max(0, y2 - y1)
            total_height += box_height + 10

        total_height += top_margin
        img_width = 1000
        final_image = np.ones((total_height, img_width, 3), dtype=np.uint8) * 255
        y_offset = top_margin

        # Place cropped images with IDs
        for i, box in enumerate(boxes_with_ids):
            x1, y1 = int(box['bounding_box'][0][0] * self.doc[0].shape[1]), int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            x2, y2 = int(box['bounding_box'][1][0] * self.doc[0].shape[1]), int(box['bounding_box'][1][1] * self.doc[0].shape[0])

            if y2 <= y1 or x2 <= x1:
                print(f"Skipping invalid bounding box for BB{i}.")
                continue

            cropped_image = self.doc[0][y1:y2, x1:x2]
            crop_height, crop_width, _ = cropped_image.shape

            if crop_height <= 0 or crop_width <= 0:
                print(f"Skipping BB{i}: Cropped image has invalid dimensions.")
                continue

            if y_offset + crop_height > final_image.shape[0]:
                print(f"Error: Insufficient space for BB{i}.")
                break

            final_image[y_offset:y_offset + crop_height, 0:crop_width] = cropped_image
            font_scale = crop_height / 50
            font_thickness = max(1, int(font_scale * 2))
            text_color = (0, 0, 0)

            text = f"{box['id']}"
            text_x = crop_width + 10
            text_y = y_offset + (crop_height // 2)

            cv2.putText(final_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            y_offset += crop_height + 10

        cv2.imwrite(output_image_path, final_image)

class VLM:
    def __init__(self, token):
        """
        Initialize the VLM class with authentication token.

        Args:
            token (str): Authentication token for Hugging Face.
        """
        login(token)

    def model(self, model_id, out_json):
        """
        Load the specified model and processor.

        Args:
            model_id (str): Model identifier.
            out_json (bool): Whether to return JSON outputs.
        """
        self.output_json = out_json
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process(self, image_path, input_text):
        """
        Process an image and text input using the loaded model.

        Args:
            image_path (str): Path to the image.
            input_text (str): Text instruction or query.

        Returns:
            str: Model output.
        """
        image = Image.open(image_path)
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        text_inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)

        inputs = {
            'input_ids': text_inputs['input_ids'],
            'pixel_values': image_inputs['pixel_values']
        }

        output = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)

        if self.output_json:
            try:
                return json.dumps(json.loads(decoded_output), indent=4)
            except json.JSONDecodeError:
                print("Invalid JSON output:", decoded_output)
        return decoded_output

    def internvl_process(self, image_path, input_text, model_id='OpenGVLab/InternVL2-8B'):
        """
        Process image and text input using InternVL pipeline.

        Args:
            image_path (str): Path to the image.
            input_text (str): Text instruction or query.
            model_id (str): Model identifier.

        Returns:
            str: Model output.
        """
        image = load_image(image_path)
        pipe = lmdeploy_pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192))
        gen_config = GenerationConfig(top_p=1, top_k=2, temperature=0)
        response = pipe((input_text, image), gen_config=gen_config)
        return response.text

class PixtralProcessor:
    def __init__(self, model_name="mistralai/Pixtral-12B-2409", max_tokens=150192):
        """
        Initialize PixtralProcessor with sampling parameters and model.

        Args:
            model_name (str): Model identifier.
            max_tokens (int): Maximum tokens for sampling.
        """
        self.model_name = model_name
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        self.llm = LLM(model=self.model_name, tokenizer_mode="mistral")

    def image_to_data_uri(self, image_path):
        """
        Convert a local image to a base64 data URI.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded data URI.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

    def pixtral(self, image_path1, image_path2, prompt):
        """
        Process multiple images and a prompt using Pixtral.

        Args:
            image_path1 (str): Path to the first image.
            image_path2 (str): Path to the second image.
            prompt (str): Text prompt for the model.

        Returns:
            str: Model output.
        """
        image_data_uri1 = self.image_to_data_uri(image_path1)
        image_data_uri2 = self.image_to_data_uri(image_path2)
        messages = [
            {
                "role": "system",
                "content": "Instructions for extracting text and bounding boxes..."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri1}},
                    {"type": "image_url", "image_url": {"url": image_data_uri2}}
                ]
            }
        ]
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
 
class Annotation:
    def __init__(self):
        """
        Initialize the Annotation class.
        """
        pass

    def flatten_bounding_box(self, bounding_box):
        """
        Flatten a bounding box if it is a nested list with a single element.

        Args:
            bounding_box (list): Bounding box coordinates.

        Returns:
            list: Flattened bounding box or the original if no flattening is needed.
        """
        if len(bounding_box) == 1 and isinstance(bounding_box[0], list):
            return bounding_box[0]
        return bounding_box

    def draw_bounding_boxes(self, image_path, json_data, output_path):
        """
        Draw bounding boxes on an image using JSON data.

        Args:
            image_path (str): Path to the input image.
            json_data (dict): JSON data containing bounding boxes and keys.
            output_path (str): Path to save the annotated image.
        """
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        for key, value in json_data.items():
            if isinstance(value, list):
                # Handle multiple bounding boxes for a key
                for box in value:
                    bounding_box = self.flatten_bounding_box(box["bounding_box"])
                    x1 = int(bounding_box[0][0] * image_width)
                    y1 = int(bounding_box[0][1] * image_height)
                    x2 = int(bounding_box[1][0] * image_width)
                    y2 = int(bounding_box[1][1] * image_height)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif "bounding_box" in value and value["bounding_box"] is not None:
                # Handle single bounding box for a key
                bounding_box = self.flatten_bounding_box(value["bounding_box"])
                x1 = int(bounding_box[0][0] * image_width)
                y1 = int(bounding_box[0][1] * image_height)
                x2 = int(bounding_box[1][0] * image_width)
                y2 = int(bounding_box[1][1] * image_height)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(output_path, image)

    def annotate_image_with_text_recognition(self, image_path, ocr_result, output_path):
        """
        Annotate an image with OCR results, including bounding boxes and text.

        Args:
            image_path (str): Path to the input image.
            ocr_result (list): OCR results containing bounding boxes and recognized text.
            output_path (str): Path to save the annotated image.
        """
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        for word_data in ocr_result:
            word = word_data['value']
            bounding_box = word_data['bounding_box']
            x1 = int(bounding_box[0][0] * image_width)
            y1 = int(bounding_box[0][1] * image_height)
            x2 = int(bounding_box[1][0] * image_width)
            y2 = int(bounding_box[1][1] * image_height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite(output_path, image)


def extract_code_block(s):
    """
    Extract the code block enclosed in triple backticks.

    Args:
        s (str): Input string.

    Returns:
        str: Extracted code block.

    Raises:
        ValueError: If no code block is found.
    """
    pattern = r'```(?:[\w]+\n)?(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        raise ValueError('No code block found in the string.')
    return match.group(1).strip()

def clean_code_block(code_block):
    """
    Remove unwanted prefixes (e.g., 'information =') from a code block.

    Args:
        code_block (str): Raw code block.

    Returns:
        str: Cleaned code block.
    """
    return code_block.replace('information =', '').strip()

def convert_tuples_to_lists(s):
    """
    Convert tuples in a string to lists and handle coordinate pairs.

    Args:
        s (str): Input string.

    Returns:
        str: Modified string with tuples converted to lists.
    """
    s = s.replace('(', '[').replace(')', ']')
    s = re.sub(r'\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]', 
               r'[[\1, \2], [\3, \4]]', s)
    return s

def sanitize_malformed_floats(s):
    """
    Correct malformed floating-point numbers in a string.

    Args:
        s (str): Input string.

    Returns:
        str: Corrected string.
    """
    return re.sub(r'(\d+\.\d+)\.\d+', r'\1', s)

def balance_brackets(s):
    """
    Ensure brackets in a string are balanced.

    Args:
        s (str): Input string.

    Returns:
        str: String with balanced brackets.

    Raises:
        ValueError: If mismatched brackets are found.
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
                    raise ValueError("Mismatched brackets found.")
            else:
                char = '}' if char == ']' else ']'
        balanced_s.append(char)

    while stack:
        last_open = stack.pop()
        balanced_s.append('}' if last_open == '{' else ']')

    return ''.join(balanced_s)

def parse_to_dict(s):
    """
    Parse a string into a Python dictionary.

    Args:
        s (str): Input string.

    Returns:
        dict: Parsed dictionary.

    Raises:
        ValueError: If there is a syntax error during parsing.
    """
    try:
        s = balance_brackets(s)
        return ast.literal_eval(s)
    except SyntaxError as e:
        raise ValueError(f"Error parsing string to dict: {e}")

def make_json_serializable(data):
    """
    Ensure data is JSON serializable.

    Args:
        data (dict): Input data.

    Returns:
        dict: JSON-serializable data.
    """
    return json.loads(json.dumps(data))

def merge_bounding_boxes(bounding_boxes):
    """
    Merge multiple bounding boxes into a single bounding box.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Merged bounding box.
    """
    if isinstance(bounding_boxes[0][0], float):
        return bounding_boxes  # Single bounding box

    x1, y1 = bounding_boxes[0][0]
    max_x = max(box[1][0] for box in bounding_boxes)
    max_y = max(box[1][1] for box in bounding_boxes)

    return [[x1, y1], [max_x, max_y]]

def correct_bounding_boxes(bounding_boxes):
    """
    Correct and flatten bounding boxes for proper structure.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Corrected bounding boxes.
    """
    corrected_boxes = []
    all_numbers = []

    for box in bounding_boxes:
        if isinstance(box, list):
            for coord_pair in box:
                all_numbers.extend(coord_pair)

    if len(all_numbers) % 4 != 0:
        return corrected_boxes

    for i in range(0, len(all_numbers), 4):
        corrected_boxes.append([[all_numbers[i], all_numbers[i + 1]], [all_numbers[i + 2], all_numbers[i + 3]]])

    return corrected_boxes
