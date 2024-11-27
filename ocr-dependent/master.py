import json
import os
from ml import OCR, VLM, Annotation, PixtralProcessor
from anls import anls_score

def main(image_path, user_queries, dataset_name):
    """
    Process an image to perform OCR, annotate it, and extract information based on user queries.

    Args:
        image_path (str): Path to the input image.
        user_queries (str): User-provided queries for information extraction.
        dataset_name (str): Dataset name (e.g., CORD, FUNSD, SROIE).

    Returns:
        dict: Extracted information and bounding boxes.
    """
    try:
        # Load configuration and authentication token
        with open("config.json", "r") as file:
            config = json.load(file)
            hf_token = config["hf_token"]

        # Extract filename details
        filename_with_extension = os.path.basename(image_path)
        filename, _ = os.path.splitext(filename_with_extension)

        # Define output filenames for results
        newoutput = f"pixtralPinakiprompt-{filename}_output8b"
        text_recognition_output_path = f"ocr-dependent/results/2-text recognition/{dataset_name}/{newoutput}.json"
        annotated_image_output_path = f"ocr-dependent/results/1-text detection/{dataset_name}/{newoutput}.png"
        extractor_output_path = f"ocr-dependent/results/3-information extraction/{dataset_name}/{newoutput}.json"

        # Ensure output directories exist
        os.makedirs(os.path.dirname(text_recognition_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(annotated_image_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(extractor_output_path), exist_ok=True)

        # Perform OCR on the image
        processor = OCR()
        processor.process(image_path)
        ocr_out = processor.get_words()  # Extract words and bounding boxes

        # Save OCR output to a JSON file
        with open(text_recognition_output_path, "w") as f:
            json.dump(ocr_out, f, indent=4)

        # Annotate the image with OCR results
        annotator = Annotation()
        annotator.annotate_image_with_text_recognition(image_path, ocr_out, annotated_image_output_path)
        print(f"Annotated image saved at {annotated_image_output_path}")

        # Initialize Vision-Language Model (VLM) for processing
        vlm = VLM(hf_token)

        # Initialize PixtralProcessor for information extraction
        pixtral_processor = PixtralProcessor()

        # Construct the prompt for the extractor model
        Extractor_prompt = f'''You are provided with an image and its bounding boxes. 

        Bounding Boxes:
        {ocr_out}

        Your task is to extract the following information along with their bounding boxes:
        {user_queries}

        Please adhere to the following guidelines:
        Ensure that all correct pairs of information and bounding boxes are included as a dictionary.
        Details should only contain the answer part.'''

        # Extract information based on the prompt
        Extractor_out = pixtral_processor.pixtral(image_path, Extractor_prompt)

        # Save the extracted information to a JSON file
        with open(extractor_output_path, "w") as f:
            json.dump(Extractor_out, f, indent=4)

        return Extractor_out

    except Exception as e:
        # Handle exceptions and log the error
        print(f"Error processing {image_path}: {e}")
        return None

def process_batch(dataset_path, dataset_name):
    """
    Process a batch of images from a dataset for OCR, annotation, and information extraction.

    Args:
        dataset_path (str): Path to the dataset JSON file.
        dataset_name (str): Dataset name (e.g., CORD, FUNSD, SROIE).
    """
    img_paths = set()
    clean_dict = {}
    total_score = 0

    # Load dataset and organize conversations by image
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    for doc in data:
        img_path = doc.get('image_path')
        if img_path:
            img_paths.add(img_path)
            clean_dict.setdefault(img_path, []).extend(doc.get('conversations', []))

    # Process each image in the dataset
    for key, value in clean_dict.items():
        try:
            # Extract instructions and responses
            instruction = [item['value'] for item in value if item['from'] == 'instruction']
            response = [item['value'] for item in value if item['from'] == 'response']

            # Process the image using the main function
            out = main(f'datasets/images/{dataset_name}/{os.path.basename(key)}', '\n '.join(instruction), dataset_name)
            if out is None:
                continue

            # Clean and parse the extractor output
            cleaned_data = out.replace('```json\n', '').replace('```', '').replace("((", '"(').replace("))", ')"').replace("\"'", "\"").replace("'\"", "\"").replace(".'", '')
            json_data = json.loads(cleaned_data)

            # Evaluate the extracted answers against the ground truth responses
            complete_answers = [item['complete_answer'] for item in json_data]
            image_score = 0
            for i in range(min(len(complete_answers), len(response))):
                image_score += anls_score(prediction=complete_answers[i], gold_labels=[response[i]], threshold=0.5)

            # Print and accumulate the score for the image
            avg_score = image_score / max(1, len(complete_answers))
            print(f"Image ANLS Score for {key}: {avg_score}")
            total_score += avg_score

        except Exception as e:
            print(f"An error occurred while processing {key}: {e}")
            continue

    # Compute and print the final average score
    final_score = total_score / max(1, len(clean_dict))
    print(f"Final Average ANLS Score: {final_score}")

if __name__ == "__main__":
    # Run the batch processing
    dataset_name = "FUNSD"  # Change to "CORD" or "SROIE" as needed
    dataset_path = f'datasets/updated-VIE_{dataset_name}.json'
    process_batch(dataset_path, dataset_name)
