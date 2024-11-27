import json
import os
from ml_ablation2 import OCR, VLM, Annotation, PixtralProcessor, extract_json_from_string, merge_bounding_boxes, correct_bounding_boxes
from anls import anls_score

# List to store image paths that encounter errors during processing
error_images = []

def main(image_path, user_queries, dataset_name):
    """
    Main function to process a given image and extract information based on user queries.
    Parameters:
    - image_path: Path to the input image.
    - user_queries: List of user queries for the image.
    - dataset_name: Name of the dataset (e.g., CORD, FUNSD, SROIE).
    """
    try:
        # Load the Hugging Face token from the configuration file
        with open("config.json", "r") as file:
            config = json.load(file)
            hf_token = config["hf_token"]

        # Extract the filename and extension
        filename_with_extension = os.path.basename(image_path)
        filename, file_extension = os.path.splitext(filename_with_extension)

        # Define output file paths
        newoutput = f"{filename}_output8b-pixtral"
        base_output_dir = f"ocr-free/results"
        text_recognition_output_path = f"{base_output_dir}/2-text recognition/{dataset_name}/ablation1/{newoutput}.json"
        word_bb_image_output_path = f"{base_output_dir}/6-word_bb_image/{dataset_name}/ablation1/{newoutput}_words.png"
        pixtral_output_path = f"{base_output_dir}/7-pixtral_output/{dataset_name}/ablation1/{newoutput}.json"
        extractor_output_path = f"{base_output_dir}/3-information extraction/{dataset_name}/ablation1/{newoutput}.json"
        mapping_output_path = f"{base_output_dir}/5-mapping/{dataset_name}/ablation1/{newoutput}.png"

        # Ensure required directories exist
        os.makedirs(os.path.dirname(text_recognition_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(word_bb_image_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(pixtral_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(extractor_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)

        # Initialize OCR processor and process the image
        processor = OCR()
        processor.process(image_path)

        # Save bounding box data to a JSON file
        bounding_boxes = processor.get_bounding_boxes()
        with open(text_recognition_output_path, "w") as f:
            json.dump(bounding_boxes, f, indent=4)

        # Generate an image showing word regions with bounding box IDs
        processor.generate_cropped_images_with_ids(word_bb_image_output_path)

        # Initialize PixtralProcessor for extracting information based on queries
        pixtral_processor = PixtralProcessor()

        # Construct a prompt for PixtralProcessor with user queries and bounding box data
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

        Output Example:
        {{"question_key": {{"value": "text", "bounding_box": [[...]]}}}}
        """

        # Run PixtralProcessor to extract answers
        Extractor_out = pixtral_processor.pixtral(image_path, word_bb_image_output_path, Extractor_prompt)
        Extractor_out_parsed = extract_json_from_string(Extractor_out)

        # Correct and merge bounding boxes if necessary
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

        # Save the extracted data with updated bounding boxes to a JSON file
        with open(extractor_output_path, "w") as f:
            json.dump(updated_extractor_out, f, indent=4)

        # Annotate the image with extracted key-value mappings
        Ann = Annotation()
        Ann.draw_bounding_boxes(image_path, updated_extractor_out, mapping_output_path)
        print(f"Annotated image with key-value mappings saved at {mapping_output_path}")

        return updated_extractor_out

    except Exception as e:
        # Handle errors and log the problematic image
        error_images.append(image_path)
        print(f"Error processing {image_path}: {e}")
        return None

def process_batch(dataset_path, dataset_name):
    """
    Processes a batch of images and evaluates their performance.
    Parameters:
    - dataset_path: Path to the dataset JSON file.
    - dataset_name: Name of the dataset (e.g., CORD, FUNSD, SROIE).
    """
    img_paths = set()
    clean_dict = {}
    new_score_data = 0
    temp_dict_data = {}
    processed_count = 0

    # Load dataset containing image paths and corresponding queries
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Organize dataset into a dictionary grouped by image path
    for doc in data:
        img_path = doc.get('image_path')
        if img_path:
            img_paths.add(img_path)
            clean_dict.setdefault(img_path, []).extend(doc.get('conversations', []))

    # Process each image and evaluate the extracted data
    for key, value in clean_dict.items():
        instruction = [item['value'] for item in value if item['from'] == 'instruction']
        try:
            out = main('datasets/images/' + dataset_name + '/' + os.path.basename(key), instruction, dataset_name)
            processed_count += 1

            # Evaluate the ANLS score for extracted data
            response = [item['value'] for item in value if item['from'] == 'response']
            data = [item['value'] for key, item in out.items()]
            total_score_data = sum(anls_score(prediction=data[i], gold_labels=[response[i]], threshold=0.5) for i in range(len(data)))
            avg_score_data = total_score_data / len(data)
            temp_dict_data[os.path.basename(key)] = avg_score_data
            new_score_data += avg_score_data

            print(f"Final Output ANLS for {key}: {avg_score_data}")
        except Exception as e:
            error_images.append(key)
            print(f"Error processing {key}: {e}")

    # Print final scores and error log
    print("\nFinal Scores for Processed Images:")
    for img, score in temp_dict_data.items():
        print(f"{img} = {score}")

    if error_images:
        print("\nImages with errors:")
        for img in error_images:
            print(img)

if __name__ == "__main__":
    # Run the batch processing with a specified dataset
    dataset_name = "CORD"  # Change to "FUNSD" or "SROIE" as needed
    dataset_path = f'datasets/updated-VIE_{dataset_name}.json'
    process_batch(dataset_path, dataset_name)
