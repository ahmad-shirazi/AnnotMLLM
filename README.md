# DLaVA: Document Language and Vision Assistant

A framework for answer localization in Document Visual Question Answering (VQA) with enhanced interpretability and trustworthiness. The system integrates OCR-dependent and OCR-free methods to localize answers precisely within document images.

---

## Overview

DLaVA introduces a novel approach for Document VQA by:
- Incorporating answer localization to enhance user trust and model interpretability.
- Supporting both OCR-dependent and OCR-free pipelines.
- Achieving state-of-the-art (SOTA) performance on Document VQA and Visual Information Extraction (VIE) tasks.

![Figure ](images/figure1.png)  
*Figure 1: Examples of Answer Annotations in CORD Dataset*

![Figure 2](images/figure2.png)  
*Figure 2: DLaVA Model Architecture for OCR-Dependent and OCR-Free Approaches*

---

## Key Features

1. **Interpretability**: Annotates document images with bounding boxes for precise answer localization.
2. **Flexibility**: Supports OCR-dependent and OCR-free pipelines.
3. **Efficiency**: Streamlined processing without complex pretraining requirements.

---

## Setup

### Dependencies

To run the project, install the required dependencies based on the models:

- **For Pixtral, Qwen, and LLaVA models**:
  ```bash
  pip install -r requirements-i.txt
  ```
 - **For InternVL and LLaMA models**:
  ```bash
  pip install -r requirements-p.txt
  ```
---

## Configuration

To configure the project, create a file named `config.json` in the root directory. The file should contain your Hugging Face token for accessing pretrained models. Here’s an example structure:

```json
{
  "huggingface_token": "your_token_here"
}
```

Replace "your_token_here" with your actual Hugging Face token.

---

## Project Structure

The repository is organized as follows:

```plaintext
DLaVA/
├── ocr_free/             # Contains scripts for the OCR-free pipeline
│   ├── run.py            # Main script to execute OCR-free tasks
│   └── ...
├── ocr_dependent/        # Contains scripts for the OCR-dependent pipeline
│   ├── run.py            # Main script to execute OCR-dependent tasks
│   └── ...
├── models/               # Pretrained vision-language models
├── datasets/             # Scripts for dataset preprocessing and loading
├── evaluation/           # Scripts for model evaluation and metrics calculation
├── requirements-i.txt    # Dependencies for Pixtral, Qwen, and LLaVA models
├── requirements-p.txt    # Dependencies for InternVL and LLaMA models
└── README.md             # Project documentation


```

## Usage

### OCR-Free Approach

Run the OCR-free pipeline using the following command:
```bash
python ocr_free/master.py
```

### OCR-Dependent Approach

Run the OCR-dependent pipeline using the following command:
```bash
python ocr_dependent/master.py 
```

### Command-Line Arguments

Both the OCR-free and OCR-dependent pipelines support additional command-line arguments for customization:

- `--model`: Specify the model to use (e.g., `Pixtral`, `InternVL`).
- `--dataset`: Specify the dataset to process (e.g., `CORD`, `FUNSD`).
- `--output_dir`: Directory where the results will be saved.

#### Example Usage

To run the OCR-dependent pipeline with the Pixtral model on the CORD dataset and save results to a specific directory:
```bash
python ocr_dependent/master.py --config config.json --model Pixtral --dataset CORD --output_dir ./results/
```


## Evaluation

Evaluate the performance of DLaVA using the provided evaluation scripts. The framework supports two key metrics:

1. **Textual Accuracy**: Measured using the Average Normalized Levenshtein Similarity (ANLS).
2. **Spatial Alignment**: Measured using Intersection over Union (IoU) for bounding box accuracy.



## Running Evaluation

To evaluate model performance, run:
```bash
python ocr-free/mAP@IOU.py
```
or 
```bash
python ocr-dependent/mAP@IOU.py
```

---

## Performance

### OCR-Dependent Results (ANLS Metric)

![Table 1](images/table1.png)  
*Table 1: OCR-Dependent Results (ANLS Metric)*

### OCR-Free Results (ANLS Metric)

![Table 2](images/table2.png)  
*Table 2: OCR-Free Results (ANLS Metric)*

---

## Supported Datasets

The DLaVA framework supports and has been tested on the following datasets:

1. **CORD**: Consolidated receipt dataset for structured information extraction.
2. **FUNSD**: Form Understanding in Noisy Scanned Documents.
3. **SROIE**: Scanned Receipt OCR and Information Extraction dataset.
4. **DocVQA**: Dataset for Document Visual Question Answering.

---

## Limitations and Future Work

### Limitations

- **IoU Challenges**: Low IoU scores for complex document layouts with overlapping elements.
- **Ambiguities**: Difficulty in resolving repeated values across fields, such as totals in receipts.

### Future Work

- **Fine-Tuning**: Improve bounding box annotations using techniques like LoRA.
- **Enhanced Reasoning**: Add spatial reasoning with positional priors and cross-field dependencies.
- **Extension**: Expand support to handle charts and graphical elements in documents.


