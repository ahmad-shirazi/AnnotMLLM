import json
import os
from ml_ablation1 import OCR, Annotation, PixtralProcessor, extract_json_from_string, merge_bounding_boxes, correct_bounding_boxes
from anls import anls_score

# List to store image paths that encounter errors
error_images = []

def extract_complete_answers(data):
    """
    Extract the 'complete_answer' field from a JSON string.
    Removes surrounding code block markers and parses the JSON.
    """
    data = data.replace("```json", "").replace("```", "").strip()
    parsed_data = json.loads(data)
    
    if isinstance(parsed_data, dict):
        return parsed_data["complete_answer"]
    elif isinstance(parsed_data, list):
        return [item["complete_answer"] for item in parsed_data][0]
    else:
        return ""

def main(image_path, user_queries, dataset_name):
    """
    Process an image to extract information based on user queries.
    Parameters:
    - image_path: Path to the input image.
    - user_queries: List of questions for the image.
    - dataset_name: Name of the dataset (e.g., CORD, FUNSD, SROIE).
    """
    try:
        # Load configuration details
        with open("config.json", "r") as file:
            config = json.load(file)
            hf_token = config["hf_token"]

        # Extract file name and set output paths
        filename_with_extension = os.path.basename(image_path)
        filename, _ = os.path.splitext(filename_with_extension)
        newoutput = f"{filename}_output8b-pixtral"

        # Define output file paths
        base_output_dir = f"ocr-free/results"
        text_recognition_output_path = f"{base_output_dir}/2-text recognition/{dataset_name}/ablation1/{newoutput}.json"
        word_bb_image_output_path = f"{base_output_dir}/6-word_bb_image/{dataset_name}/ablation1/{newoutput}_words.png"
        pixtral_output_path = f"{base_output_dir}/7-pixtral_output/{dataset_name}/ablation1/{newoutput}.json"
        extractor_output_path = f"{base_output_dir}/3-information extraction/{dataset_name}/ablation1/{newoutput}.json"
        mapping_output_path = f"{base_output_dir}/5-mapping/{dataset_name}/ablation1/{newoutput}.png"

        # Ensure output directories exist
        os.makedirs(os.path.dirname(text_recognition_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(word_bb_image_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(pixtral_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(extractor_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)

        # Initialize and process OCR
        processor = OCR()
        processor.process(image_path)

        # Extract bounding boxes and save them
        bounding_boxes = processor.get_bounding_boxes()
        with open(text_recognition_output_path, "w") as f:
            json.dump(bounding_boxes, f, indent=4)

        # Generate an image showing word regions with bounding box IDs
        processor.generate_cropped_images_with_ids(word_bb_image_output_path)

        # Initialize PixtralProcessor and query answers
        pixtral_processor = PixtralProcessor()
        query_answers = {}
        
        # Process each user query
        for query in user_queries:
            prompt = f"""
            You are provided with an image and a question: {query}
            Return only the extracted information without extra text.
            """
            pixtral_output = pixtral_processor.pixtral_2(image_path, prompt)
            pixtral_output_completeAns = extract_complete_answers(pixtral_output)
            query_answers[query] = pixtral_output_completeAns

        # Save Pixtral results
        with open(pixtral_output_path, "w") as f:
            json.dump(query_answers, f, indent=4)

        # Extract additional key-value information
        Extractor_prompt = f"""
        You are provided with:
        - A set of questions.
        - An input image with answers to the questions.
        - A JSON file mapping Bounding Box IDs (BB0, BB1, etc.) to their coordinates.
        - A second image showing all words with their Bounding Box IDs.

        Questions:
        {user_queries}

        Bounding Box Data:
        {bounding_boxes}

        Task:
        - Answer questions strictly using words from the input image.
        - For each question, return a JSON with:
        - "value": Answer text (limit 4-5 words; only from the image).
        - "bounding_box": Coordinates for each word in [[x1, y1], [x2, y2]] format.

        JSON Rules:
        - Do NOT include "Final Answer:" or extra text.
        - Maintain correct bracket pairs (){{}}[].
        - Use:
        - Single bounding box: "bounding_box": [[x1, y1], [x2, y2]]
        - Multiple words: "bounding_box": [[[x1, y1], [x2, y2]], ...]

        Output Example:
        {{"question_key": {{"value": "text", "bounding_box": [[...]]}}}}

        Output only the JSON. No additional text or explanations.
        """
        Extractor_out = pixtral_processor.pixtral(image_path, word_bb_image_output_path, Extractor_prompt)
        Extractor_out_parsed = extract_json_from_string(Extractor_out)

        # Correct bounding boxes and merge if necessary
        updated_extractor_out = {}
        for key, value in Extractor_out_parsed.items():
            new_key = key
            if "bounding_box" in value:
                bounding_boxes = value["bounding_box"]
                bounding_boxes = correct_bounding_boxes(bounding_boxes)
                if isinstance(bounding_boxes, list) and len(bounding_boxes) > 1:
                    merged_box = merge_bounding_boxes(bounding_boxes)
                    value["bounding_box"] = merged_box
            updated_extractor_out[new_key] = value

        # Save updated extractor output
        with open(extractor_output_path, "w") as f:
            json.dump(updated_extractor_out, f, indent=4)

        # Annotate and save bounding boxes on the image
        Ann = Annotation()
        Ann.draw_bounding_boxes(image_path, updated_extractor_out, mapping_output_path)
        return query_answers, updated_extractor_out

    except Exception as e:
        error_images.append(image_path)
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_batch(dataset_path, dataset_name):
    """
    Batch process a dataset to evaluate and extract information.
    Parameters:
    - dataset_path: Path to the dataset JSON file.
    - dataset_name: Name of the dataset (e.g., CORD, FUNSD, SROIE).
    """
    img_paths = set()
    clean_dict = {}
    new_score_data = 0
    new_score_qa = 0
    temp_dict_data = {}
    temp_dict_qa = {}
    processed_count = 0

    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Organize dataset by image path
    for doc in data:
        img_path = doc.get('image_path')
        if img_path:
            img_paths.add(img_path)
            clean_dict.setdefault(img_path, []).extend(doc.get('conversations', []))

    # Process each image in the dataset
    for key, value in clean_dict.items():
        instruction = [item['value'] for item in value if item['from'] == 'instruction']
        try:
            Query_Ans, out = main(f'datasets/images/{dataset_name}/' + os.path.basename(key), instruction, dataset_name)
            if Query_Ans is None or out is None:
                continue
            processed_count += 1
        except Exception as e:
            error_images.append(key)
            print(f"Error processing {key}: {e}")

    # Print errors and final scores
    if error_images:
        print("\nImages that encountered errors:")
        for img in error_images:
            print(img)

if __name__ == "__main__":
    # Run the batch processing
    dataset_name = "CORD"  # Change to "FUNSD" or "SROIE" for other datasets
    dataset_path = f'datasets/updated-VIE_{dataset_name}.json'
    process_batch(dataset_path, dataset_name)