# r09944076 - Applied Deep Learning Homework 2

This project implements advanced deep learning tasks as part of the "Applied Deep Learning" course. The task focuses on **Chinese Question Answering** and involves training two models:

1. **Multiple Choice Model**:
   - Identifies the relevant article from a dataset that corresponds to a given question and its answer.
   
2. **Question Answering Model**:
   - Extracts the specific location of the answer within the relevant article, enabling accurate and detailed responses.

These tasks leverage tokenizers, custom models, and utility scripts for efficient training, testing, and evaluation.

---

## Features
- Chinese Question Answering tasks with:
  - A Multiple Choice model for article selection.
  - A Question Answering model for answer span extraction.
- Efficient data handling and preprocessing scripts (`src/data.py`).
- Easy-to-use training and testing workflows.
- Predefined environment setup for consistency.

---

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Conda or a similar environment manager.

### Setup
1. Create and activate a Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate adl-hw2
   ```

2. Install additional requirements (if needed):
   ```bash
   pip install -r requirements.in
   ```

---

## Usage

### Train the Models

#### Multiple Choice Model
To train the multiple choice model:
```bash
python3 mc_bert_train.py \
    --context=<path_to_json_of_context> \
    --train_data=<path_to_json_file_of_train_data> \
    --val_data=<path_to_json_file_of_validation_data> \
    --mc_model_name=<BERT_model_name_for_multiple_choices> \
    --max_q_len=<the_max_length_of_question> \
    --max_p_len=<the_max_length_of_paragraph> \
    --max_s_len=<the_max_length_of_sequence> \
    --learning_rate=<learning_rate_of_optimization> \
    --weight_decay=<the_value_of_weight_decay> \
    --warmup_division=<the_percentage_of_warmup> \
    --batchsize=<batch_size> \
    --iterNum=<the_number_of_gradient_accumulation> \
    --num_workers=<the_number_of_workers_for_dataloader> \
    --epoch=<the_number_of_epoch> \
    --model_dir=<directory_for_saving_model>
```

#### Question Answering Model
To train the question answering model:
```bash
python3 src/qa_bert_train.py \
    --context=<path_to_json_of_context> \
    --train_data=<path_to_json_file_of_train_data> \
    --val_data=<path_to_json_file_of_validation_data> \
    --qa_model_name=<BERT_model_name_for_question_answering> \
    --max_q_len=<the_max_length_of_question> \
    --max_p_len=<the_max_length_of_paragraph> \
    --max_s_len=<the_max_length_of_sequence> \
    --window_interval=<the_size_of_paragraph_interval> \
    --learning_rate=<learning_rate_of_optimization> \
    --weight_decay=<the_value_of_weight_decay> \
    --warmup_division=<the_percentage_of_warmup> \
    --batchsize=<batch_size> \
    --iterNum=<the_number_of_gradient_accumulation> \
    --num_workers=<the_number_of_workers_for_dataloader> \
    --epoch=<the_number_of_epoch> \
    --model_dir=<directory_for_saving_model>
```

### Test the Models

python3 test.py \
    --context=<path_to_json_of_context> \
    --test_data=<path_to_json_file_of_test_data> \
    --max_q_len=<the_max_length_of_question> \
    --max_p_len=<the_max_length_of_paragraph> \
    --max_s_len=<the_max_length_of_sequence> \
    --window_interval=<the_size_of_paragraph_interval> \
    --num_workers=<the_number_of_workers_for_dataloader> \
    --epoch=<the_number_of_epoch> \
    --model_dir=<directory_for_saving_model> \
    --predict=<the_output_of_testing_data>

---

## Project Structure
- **`src/model.py`**: Defines the deep learning models for multiple choice and question answering tasks.
- **`src/train_multiple_choice.py`**: Training script for the multiple choice model.
- **`src/train_question_answering.py`**: Training script for the question answering model.
- **`src/test_multiple_choice.py`**: Testing script for the multiple choice model.
- **`src/test_question_answering.py`**: Testing script for the question answering model.
- **`src/data.py`**: Data preprocessing and loading utilities.
- **`src/utils.py`**: Helper functions for common tasks (e.g., saving/loading models, printing metrics).
- **`preprocess.sh`**: Preprocessing script to prepare data for training and testing.
- **`environment.yml`**: Conda environment configuration file.
- **`requirements.in`**: Python dependencies.
- **`README.md`**: This file, describing the project and its usage.

---

## License
This project is for educational purposes as part of the Applied Deep Learning course. Please maintain academic integrity when using this repository.

---

## Acknowledgments
Special thanks to the teaching team of Applied Deep Learning for their guidance and support in this assignment.

---
"""