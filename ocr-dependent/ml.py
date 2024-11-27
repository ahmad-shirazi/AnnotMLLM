import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import cv2
import numpy as np
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from transformers import pipeline as transformers_pipeline
from vllm import LLM
from vllm.sampling_params import SamplingParams
import base64

# OCR class for performing text detection and recognition on images
class OCR:
    def __init__(self):
        """
        Initialize the OCR predictor with specific detection and recognition architectures.
        """
        self.predictor = ocr_predictor(
            det_arch='db_resnet50',  # Detection architecture
            reco_arch='parseq',     # Recognition architecture
            pretrained=True,        # Use pretrained model
            assume_straight_pages=True  # Assume straight pages for input documents
        )

    def process(self, file_path):
        """
        Process an input image file to detect and recognize text.

        Args:
            file_path (str): Path to the input image file.

        Returns:
            result (dict): Output of OCR processing.
        """
        self.doc = DocumentFile.from_images(file_path)  # Load image as a document
        self.result = self.predictor(self.doc)          # Perform OCR on the document
        return self.result

    def get_words(self):
        """
        Extract words and their bounding boxes from OCR results.

        Returns:
            list: List of dictionaries with word values and bounding boxes.
        """
        self.OCRout = self.result.export()  # Export OCR results
        words = [
            {'value': word['value'], 'bounding_box': word['geometry']}
            for block in self.OCRout['pages'][0]['blocks']
            for line in block['lines']
            for word in line['words']
        ]
        return words

# VLM class for Vision-Language Model processing
class VLM:
    def __init__(self, token):
        """
        Initialize the VLM class and authenticate with HuggingFace Hub.

        Args:
            token (str): HuggingFace API token.
        """
        login(token)

    def model(self, model_id, out_json):
        """
        Load the specified model and its processor.

        Args:
            model_id (str): ID of the pretrained model to load.
            out_json (bool): Whether to output results in JSON format.
        """
        self.output_json = out_json
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
            device_map="auto"           # Automatically map the model to devices
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def Process(self, image_path, input_text):
        """
        Process an image and text input using the Vision-Language Model.

        Args:
            image_path (str): Path to the input image.
            input_text (str): Input text prompt.

        Returns:
            str: Decoded output from the model.
        """
        image = Image.open(image_path)  # Load the image
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        text_inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)

        # Combine image and text inputs for model inference
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'pixel_values': image_inputs['pixel_values']
        }

        # Generate output
        output = self.model.generate(
            **inputs,
            max_new_tokens=8192,  # Maximum number of tokens in the output
            do_sample=False       # Disable sampling for deterministic output
        )

        # Decode the output and optionally format it as JSON
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        if self.output_json:
            try:
                output_json = json.loads(decoded_output)
                return json.dumps(output_json, indent=4)
            except json.JSONDecodeError:
                print("The model output is not a valid JSON:", decoded_output)
        else:
            return decoded_output

    def internvl_process(self, image_path, input_text, model_id='OpenGVLab/InternVL2-8B'):
        """
        Process an image and text input using the InternVL model.

        Args:
            image_path (str): Path to the input image.
            input_text (str): Input text prompt.
            model_id (str): Model ID for the InternVL pipeline.

        Returns:
            str: Generated response from the model.
        """
        image = load_image(image_path)  # Load the image
        pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192))
        gen_config = GenerationConfig(top_p=1, top_k=2, temperature=0)  # Generation parameters

        input_data = (input_text, image)  # Prepare input data
        response = pipe(input_data, gen_config=gen_config)  # Generate response
        return response.text

    def format_output_as_json(self, output):
        """
        Format the model's output into key-value pairs with bounding boxes as JSON.

        Args:
            output (list): Raw output from the model.

        Returns:
            str: JSON-formatted output.
        """
        formatted_output = []
        id_counter = 0

        for message in output[0]["generated_text"]:
            if message["role"] == "assistant":
                content = message["content"]
                lines = content.split("\n")  # Split content into lines
                for i in range(len(lines)):
                    if "Value" in lines[i]:
                        key = lines[i - 1].strip(": ")
                        value = lines[i].split(": ")[1].strip()
                        bounding_boxes = []
                        j = i + 1

                        # Extract bounding box coordinates
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

# PixtralProcessor class for advanced processing using Pixtral models
class PixtralProcessor:
    def __init__(self, model_name="mistralai/Pixtral-12B-2409", max_tokens=48192):
        """
        Initialize the PixtralProcessor with a specific model and token limit.

        Args:
            model_name (str): Name of the Pixtral model.
            max_tokens (int): Maximum number of tokens for output.
        """
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens)
        self.llm = LLM(model=self.model_name, tokenizer_mode="mistral")

    def image_to_data_uri(self, image_path):
        """
        Convert an image file to a Base64 data URI.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Base64 data URI of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

    def pixtral(self, image_path, prompt):
        """
        Process an image with a prompt using the Pixtral model.

        Args:
            image_path (str): Path to the input image.
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated response from the model.
        """
        image_data_uri = self.image_to_data_uri(image_path)
        messages = [
            {"role": "system", "content": "..."},  # System prompt
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]}
        ]
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
