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
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import numpy as np
import ast
import json
import re

class OCR:
    def __init__(self):
        """
        Initialize the OCR class with a pre-trained OCR predictor.
        """
        self.predictor = ocr_predictor(det_arch='db_resnet50', pretrained=True, assume_straight_pages=True)

    def process(self, file_path):
        """
        Process an image or document to perform OCR.

        Args:
            file_path (str): Path to the input image or document file.

        Returns:
            dict: OCR results.
        """
        self.doc = DocumentFile.from_images(file_path)
        self.result = self.predictor(self.doc)
        return self.result

    def get_bounding_boxes(self):
        """
        Extract bounding boxes from the OCR results and assign unique IDs.

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
                        'bounding_box': word['geometry']  # Only include bounding box coordinates
                    })
                    box_id += 1
        return boxes_with_ids

    def generate_cropped_images_with_ids(self, output_image_path):
        """
        Generate a single image containing cropped regions of bounding boxes
        with their corresponding IDs displayed next to them.

        Args:
            output_image_path (str): Path to save the final output image.
        """
        boxes_with_ids = self.get_bounding_boxes()

        # Calculate the total height needed for the final image
        top_margin = 20
        total_height = 0
        for box in boxes_with_ids:
            y1 = int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            y2 = int(box['bounding_box'][1][1] * self.doc[0].shape[0])
            box_height = max(0, y2 - y1)
            total_height += box_height + 10  # Add padding between crops

        total_height += top_margin

        # Define the width of the final image
        img_width = 1000
        final_image = np.ones((total_height, img_width, 3), dtype=np.uint8) * 255  # Create a white background

        y_offset = top_margin

        # Process each bounding box
        for i, box in enumerate(boxes_with_ids):
            x1, y1 = int(box['bounding_box'][0][0] * self.doc[0].shape[1]), int(box['bounding_box'][0][1] * self.doc[0].shape[0])
            x2, y2 = int(box['bounding_box'][1][0] * self.doc[0].shape[1]), int(box['bounding_box'][1][1] * self.doc[0].shape[0])

            # Skip invalid bounding boxes
            if y2 <= y1 or x2 <= x1:
                print(f"Skipping invalid bounding box for BB{i}: height or width is zero or negative.")
                continue

            # Extract the cropped image
            cropped_image = self.doc[0][y1:y2, x1:x2]
            crop_height, crop_width, _ = cropped_image.shape

            # Skip if cropped image dimensions are invalid
            if crop_height <= 0 or crop_width <= 0:
                print(f"Skipping BB{i}: Cropped image has zero or negative dimensions.")
                continue

            # Check if there is enough space in the final image
            if y_offset + crop_height > final_image.shape[0]:
                print(f"Error: Insufficient space in final image to place cropped image for BB{i}.")
                break

            # Place the cropped image on the final image
            final_image[y_offset:y_offset + crop_height, 0:crop_width] = cropped_image

            # Add the bounding box ID next to the cropped image
            font_scale = crop_height / 50  # Dynamically adjust text size
            font_thickness = max(1, int(font_scale * 2))
            text_color = (0, 0, 0)  # Black text
            text = f"{box['id']}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = crop_width + 10
            text_y = y_offset + (crop_height // 2) + (text_size[1] // 2)  # Align text with the middle of the cropped image

            # Draw the bounding box ID
            cv2.putText(final_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

            # Update the vertical offset for the next cropped image
            y_offset += crop_height + 10

        # Save the final image
        cv2.imwrite(output_image_path, final_image)


class VLM:
    def __init__(self, token):
        """
        Initialize the VLM class with authentication using a token.

        Args:
            token (str): Hugging Face Hub token for authentication.
        """
        login(token)

    def model(self, model_id, out_json):
        """
        Load the specified multimodal model and processor.

        Args:
            model_id (str): The model identifier from Hugging Face Hub.
            out_json (bool): Whether to format outputs as JSON.
        """
        self.output_json = out_json
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def Process(self, image_path, input_text):
        """
        Process an image and input text using the loaded multimodal model.

        Args:
            image_path (str): Path to the image file.
            input_text (str): Instruction or query for the model.

        Returns:
            str: The model's output, formatted as JSON if specified.
        """
        # Load the image
        image = Image.open(image_path)

        # Prepare the image and text inputs
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        text_inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)

        # Combine image and text inputs for model generation
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'pixel_values': image_inputs['pixel_values']
        }

        # Generate output from the model
        output = self.model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False  # Deterministic output
        )

        # Decode the output
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)

        if self.output_json:
            # Parse output as JSON if specified
            try:
                output_json = json.loads(decoded_output)
                return json.dumps(output_json, indent=4)
            except json.JSONDecodeError:
                print("The model output is not valid JSON:", decoded_output)
        else:
            return decoded_output

    def internvl_process(self, image_path, input_text, model_id='OpenGVLab/InternVL2-8B'):
        """
        Process an image and input text using the InternVL pipeline.

        Args:
            image_path (str): Path to the image file.
            input_text (str): Instruction or query for the model.
            model_id (str): Identifier of the InternVL model.

        Returns:
            str: The text response from the pipeline.
        """
        # Load the image
        image = load_image(image_path)

        # Set up the InternVL pipeline
        pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192))

        # Configure generation settings
        gen_config = GenerationConfig(top_p=1, top_k=2, temperature=0)

        # Generate the response
        response = pipe((input_text, image), gen_config=gen_config)

        return response.text

    def format_output_as_json(self, output):
        """
        Format the model's output into a structured JSON format.

        Args:
            output (list): Model-generated output.

        Returns:
            str: Formatted JSON string containing keys, values, bounding boxes, and IDs.
        """
        formatted_output = []
        id_counter = 0

        for message in output[0]["generated_text"]:
            if message["role"] == "assistant":
                content = message["content"]
                lines = content.split("\n")
                for i in range(len(lines)):
                    if "Value" in lines[i]:
                        key = lines[i - 1].strip(": ")
                        value = lines[i].split(": ")[1].strip()
                        bounding_boxes = []

                        # Collect bounding box coordinates
                        j = i + 1
                        while j < len(lines) and "bounding_box" in lines[j]:
                            bounding_box_match = re.search(r'bounding_box\': \(\((.*?)\)\)', lines[j])
                            if bounding_box_match:
                                bounding_boxes.append(eval(f"(({bounding_box_match.group(1)}))"))
                            j += 1

                        bounding_box_value = bounding_boxes if bounding_boxes else None

                        formatted_output.append({
                            "Key": key,
                            "Value": value,
                            "Bounding Box": bounding_box_value,
                            "id": id_counter
                        })
                        id_counter += 1

        return json.dumps(formatted_output, indent=4)

    def LLM(self, input_text, model_name='meta-llama/Meta-Llama-3.1-8B-Instruct'):
        """
        Process input text using a text generation pipeline.

        Args:
            input_text (str): The instruction or query for the LLM.
            model_name (str): Identifier of the LLM model.

        Returns:
            list: Generated output from the LLM.
        """
        # Initialize the text-generation pipeline
        llm_pipeline = transformers_pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # Define messages for the model
        messages = [
            {"role": "system", "content": '''You will follow instructions precisely. You only answer the questions and answers that the user asks, formatted exclusively in JSON. Here is an example of the expected output:
[
  {
    "question": "first question",
    "complete_answer": "CHEESE",
    "details": {
      "word": "CHEESE",
      "bounding_box": "(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)"
    }
  }
]'''}, 
            {"role": "user", "content": input_text},
        ]

        # Generate the response
        outputs = llm_pipeline(messages, max_new_tokens=1524, temperature=0.1)

        return outputs


class PixtralProcessor:
    def __init__(self, model_name="mistralai/Pixtral-12B-2409", max_tokens=150192):
        """
        Initialize the PixtralProcessor with a model and sampling parameters.

        Args:
            model_name (str): The name of the Pixtral model to load.
            max_tokens (int): Maximum number of tokens for the sampling parameters.
        """
        self.model_name = model_name
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        self.llm = LLM(model=self.model_name, tokenizer_mode="mistral")

    def image_to_data_uri(self, image_path):
        """
        Convert a local image file into a data URI.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded data URI of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

    def pixtral(self, image_path, prompt):
        """
        Process a single image with a prompt using Pixtral.

        Args:
            image_path (str): Path to the image file.
            prompt (str): Text prompt to guide the model.

        Returns:
            str: Model's response text.
        """
        # Convert image to data URI
        image_data_uri = self.image_to_data_uri(image_path)

        # Construct messages for the model
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI that correctly identifies bounding box coordinates "
                    "of answers to questions. You map words to their corresponding bounding boxes "
                    "based on shared bounding box IDs from two different sources."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]

        # Send request to the LLM
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def pixtral_4_inputs(self, image_path_1, image_path_2, bounding_boxes, prompt):
        """
        Process two images, bounding boxes, and a prompt using Pixtral.

        Args:
            image_path_1 (str): Path to the first image.
            image_path_2 (str): Path to the second image.
            bounding_boxes (dict): Dictionary containing bounding box data.
            prompt (str): Text prompt to guide the model.

        Returns:
            str: Model's response text.
        """
        # Convert images to data URIs
        image_data_uri_1 = self.image_to_data_uri(image_path_1)
        image_data_uri_2 = self.image_to_data_uri(image_path_2)

        # Construct messages for the model
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI that identifies bounding box coordinates based on "
                    "specific keys in the image and maps them to words using bounding box IDs."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri_1}},
                    {"type": "image_url", "image_url": {"url": image_data_uri_2}},
                    {"type": "text", "text": json.dumps(bounding_boxes)}
                ]
            }
        ]

        # Send request to the LLM
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def pixtral_2(self, image_path, prompt):
        """
        Process a single image and prompt, returning structured JSON output.

        Args:
            image_path (str): Path to the image file.
            prompt (str): Text prompt to guide the model.

        Returns:
            str: Model's response in JSON format.
        """
        # Convert image to data URI
        image_data_uri = self.image_to_data_uri(image_path)

        # Construct messages for the model
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a question-answering agent that outputs structurally correct JSON. "
                    "Answer questions accurately and format responses exclusively in JSON."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]

        # Send request to the LLM
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def process_output(self, llm_output):
        """
        Parse and structure LLM output into key-value-bounding box format.

        Args:
            llm_output (str): Raw output from the LLM.

        Returns:
            list: List of dictionaries with keys, values, and bounding boxes.
        """
        structured_output = []

        # Split output into lines
        lines = llm_output.split('\n')
        current_key, current_value, current_bounding_boxes = "", "", []

        for line in lines:
            # Extract key and value from lines
            if "Name of Menu" in line or "Total Price of Menu" in line:
                if current_key and current_value:
                    # Store previous entry
                    structured_output.append({
                        "key": current_key,
                        "value": current_value,
                        "bounding_box": current_bounding_boxes
                    })
                # Reset for new entry
                current_key, current_value, current_bounding_boxes = line.strip(), "", []
            elif line.startswith("-"):
                if not current_value:
                    current_value = line.split(":")[1].strip()
                else:
                    # Add bounding box coordinates
                    bounding_box = self.extract_bounding_box_from_line(line)
                    current_bounding_boxes.append(bounding_box)

        # Add last entry
        if current_key and current_value:
            structured_output.append({
                "key": current_key,
                "value": current_value,
                "bounding_box": current_bounding_boxes
            })

        return structured_output

    def extract_bounding_box_from_line(self, line):
        """
        Extract bounding box coordinates from a line of text.

        Args:
            line (str): Line containing bounding box information.

        Returns:
            tuple: Extracted bounding box coordinates.
        """
        bounding_box_part = line.split("'bounding_box':")[-1].strip()
        return eval(bounding_box_part)


    

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
            bounding_box (list): A list representing bounding box coordinates.

        Returns:
            list: Flattened bounding box if applicable, otherwise the original.
        """
        if len(bounding_box) == 1 and isinstance(bounding_box[0], list):
            return bounding_box[0]
        return bounding_box

    def draw_bounding_boxes(self, image_path, json_data, output_path):
        """
        Draw bounding boxes on an image based on JSON data.

        Args:
            image_path (str): Path to the input image.
            json_data (dict): JSON data containing bounding box coordinates and labels.
            output_path (str): Path to save the annotated image.
        """
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        for key, value in json_data.items():
            if isinstance(value, list):
                # Handle multiple bounding boxes for a single key
                for box in value:
                    bounding_box = self.flatten_bounding_box(box["bounding_box"])
                    x1 = int(bounding_box[0][0] * image_width)
                    y1 = int(bounding_box[0][1] * image_height)
                    x2 = int(bounding_box[1][0] * image_width)
                    y2 = int(bounding_box[1][1] * image_height)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif "bounding_box" in value and value["bounding_box"] is not None:
                # Handle a single bounding box
                bounding_box = self.flatten_bounding_box(value["bounding_box"])
                x1 = int(bounding_box[0][0] * image_width)
                y1 = int(bounding_box[0][1] * image_height)
                x2 = int(bounding_box[1][0] * image_width)
                y2 = int(bounding_box[1][1] * image_height)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the annotated image
        cv2.imwrite(output_path, image)

    def annotate_image_with_text_recognition(self, image_path, ocr_result, output_path):
        """
        Annotate an image with bounding boxes and recognized text from OCR results.

        Args:
            image_path (str): Path to the input image.
            ocr_result (list): List of OCR results containing text and bounding boxes.
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

        # Save the annotated image
        cv2.imwrite(output_path, image)


########################### Functions for Converting String to JSON ################################

def extract_code_block(s):
    """
    Extract a code block enclosed in triple backticks.

    Args:
        s (str): Input string containing code.

    Returns:
        str: Extracted code block.
    """
    pattern = r'```(?:[\w]+\n)?(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        raise ValueError('No code block found in the string.')
    return match.group(1).strip()


def clean_code_block(code_block):
    """
    Clean a code block by removing specific prefixes.

    Args:
        code_block (str): Raw code block.

    Returns:
        str: Cleaned code block.
    """
    return code_block.replace('information =', '').strip()


def convert_tuples_to_lists(s):
    """
    Convert tuples in a string to lists and group numbers into coordinate pairs.

    Args:
        s (str): String containing tuples.

    Returns:
        str: String with tuples converted to lists.
    """
    s = s.replace('(', '[').replace(')', ']')
    s = re.sub(r'\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]',
               r'[[\1, \2], [\3, \4]]', s)
    return s


def parse_to_dict(s):
    """
    Parse a string into a dictionary.

    Args:
        s (str): Input string.

    Returns:
        dict: Parsed dictionary.
    """
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Error parsing string to dict: {e}")


def make_json_serializable(data):
    """
    Ensure a dictionary is JSON serializable.

    Args:
        data (dict): Input dictionary.

    Returns:
        dict: JSON-serializable dictionary.
    """
    return json.loads(json.dumps(data))


def extract_json_from_string(s):
    """
    Extract and clean JSON data from a string.

    Args:
        s (str): Input string.

    Returns:
        dict: JSON data.
    """
    code_block = extract_code_block(s)
    code_block = clean_code_block(code_block)
    code_block = convert_tuples_to_lists(code_block)
    data = parse_to_dict(code_block)
    return make_json_serializable(data)


def merge_bounding_boxes(bounding_boxes):
    """
    Merge multiple bounding boxes into a single bounding box.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Merged bounding box.
    """
    if isinstance(bounding_boxes[0][0], float) and isinstance(bounding_boxes[0][1], float):
        return bounding_boxes  # Single bounding box
    if not bounding_boxes or len(bounding_boxes) < 2:
        return bounding_boxes  # No merge needed

    x1, y1 = bounding_boxes[0][0]
    max_x = max(box[1][0] for box in bounding_boxes)
    max_y = max(box[1][1] for box in bounding_boxes)

    return [[x1, y1], [max_x, max_y]]


def correct_bounding_boxes(bounding_boxes):
    """
    Correct nested or malformed bounding boxes.

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

    for i in range(0, len(all_numbers), 4):
        corrected_boxes.append([[all_numbers[i], all_numbers[i + 1]], [all_numbers[i + 2], all_numbers[i + 3]]])

    return corrected_boxes


# Function to extract the code block within triple backticks
def extract_code_block(s):
    """
    Extract a code block enclosed in triple backticks.

    Args:
        s (str): The input string containing the code block.

    Returns:
        str: The extracted code block.
    
    Raises:
        ValueError: If no code block is found.
    """
    pattern = r'```(?:[\w]+\n)?(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        raise ValueError('No code block found in the string.')
    return match.group(1).strip()


# Function to clean the code block (remove specific prefixes)
def clean_code_block(code_block):
    """
    Clean a code block by removing unwanted prefixes.

    Args:
        code_block (str): Raw code block.

    Returns:
        str: Cleaned code block.
    """
    return code_block.replace('information =', '').strip()


# Function to convert tuples to lists and handle floating-point number pairs
def convert_tuples_to_lists(s):
    """
    Convert tuples in a string to lists and group numbers into coordinate pairs.

    Args:
        s (str): Input string containing tuples.

    Returns:
        str: Modified string with tuples converted to lists.
    """
    s = s.replace('(', '[').replace(')', ']')
    s = re.sub(r'\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]', 
               r'[[\1, \2], [\3, \4]]', s)
    return s


# Function to sanitize malformed floating-point numbers
def sanitize_malformed_floats(s):
    """
    Fix malformed floating-point numbers in a string (e.g., 0.560.5 -> 0.560).

    Args:
        s (str): Input string.

    Returns:
        str: Corrected string.
    """
    return re.sub(r'(\d+\.\d+)\.\d+', r'\1', s)


# Function to ensure brackets are balanced
def balance_brackets(s):
    """
    Balance brackets in a string by adding missing opening or closing brackets.

    Args:
        s (str): Input string.

    Returns:
        str: Modified string with balanced brackets.
    
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
    Parse a string into a dictionary, ensuring balanced brackets.

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


# Function to make data JSON serializable
def make_json_serializable(data):
    """
    Ensure the parsed data is JSON serializable.

    Args:
        data (dict): Input data.

    Returns:
        dict: JSON-serializable data.
    """
    return json.loads(json.dumps(data))


# Helper function to remove quotes around bounding box coordinates
def remove_quotes_in_bounding_box(data):
    """
    Remove quotes around coordinates inside bounding boxes.

    Args:
        data (dict or list): Input data containing bounding boxes.

    Returns:
        dict or list: Modified data.
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


# Function to extract and clean JSON data from a string
def extract_json_from_string(s):
    """
    Extract, clean, and convert JSON data from a string.

    Args:
        s (str): Input string.

    Returns:
        dict: Extracted JSON data.
    """
    code_block = extract_code_block(s)
    code_block = clean_code_block(code_block)
    code_block = convert_tuples_to_lists(code_block)
    code_block = sanitize_malformed_floats(code_block)
    code_block = balance_brackets(code_block)
    data = parse_to_dict(code_block)
    data = remove_quotes_in_bounding_box(data)
    return make_json_serializable(data)


# Function to merge multiple bounding boxes into one
def merge_bounding_boxes(bounding_boxes):
    """
    Merge multiple bounding boxes into a single bounding box.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: Merged bounding box.
    """
    if isinstance(bounding_boxes[0][0], float) and isinstance(bounding_boxes[0][1], float):
        return bounding_boxes  # Single bounding box
    if not bounding_boxes or len(bounding_boxes) < 2:
        return bounding_boxes  # No merge needed

    x1, y1 = bounding_boxes[0][0]
    max_x = max(box[1][0] for box in bounding_boxes)
    max_y = max(box[1][1] for box in bounding_boxes)

    return [[x1, y1], [max_x, max_y]]


# Function to correct and flatten bounding boxes
def correct_bounding_boxes(bounding_boxes):
    """
    Correct malformed bounding boxes and ensure proper structure.

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

    for i in range(0, len(all_numbers), 4):
        corrected_boxes.append([[all_numbers[i], all_numbers[i + 1]], [all_numbers[i + 2], all_numbers[i + 3]]])

    return corrected_boxes
