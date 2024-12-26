
# r09944076 - Applied Deep Learning Homework 1

This project contains the implementation of **intent classification** and **slot tagging** tasks for natural language processing (NLP), as part of the "Applied Deep Learning" course assignments. The tasks utilize tokenizers, custom models, and utility scripts to train, test, and evaluate performance.


## Features
- Intent classification using custom models (`p1_model.py`).
- Slot tagging for sequence labeling tasks (`p2_model.py`).
- Data preprocessing and utilities for efficient model training.
- Easy-to-run scripts for training, testing, and evaluation.
- Predefined environment setup for compatibility.

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
   conda activate adl-hw1
   ```

2. Install additional requirements (if needed):
   ```bash
   pip install -r requirements.in
   ```

---

## Usage
### Preprocess
Run the preprocess code provided by TAs:
```bash
bash preprocess.sh
```

### Intent Classification
To train and test the intent classification model:
1. Train the model:
   ```bash
      python3 src/p1_train_intent.py \
      --train_data=<path_to_json_of_training_data> \
      --val_data=<path_to_json_file_of_validation_data> \
      --label_idx=<path_to_the_json_file_of_labelindex> \
      --word_vector=<path_to_the_word_embedding_data> \
      --vocab=<path_to_the_Class_Vocab_containing_the_token_information> \
      --sequence_len=<max_length_of_sequence_tokens> \
      --learning_rate=<learning_rate_of_optimization> \
      --batchsize=<batch_size> \
      --num_workers=<the_number_of_workers_for_dataloader> \
      --epoch=<the_number_of_epoch> \
      --model_dir=<directory_for_saving_model>
   ```
2. Test the model:
   ```bash
      python3 src/p1_test_intent.py \
      --test_data=<path_to_json_of_test_data> \
      --label_idx=<path_to_the_json_file_of_labelindex> \
      --word_vector=<path_to_the_word_embedding_data> \
      --vocab=<path_to_the_Class_Vocab_containing_the_token_information> \
      --sequence_len=<max_length_of_sequence_tokens> \
      --num_workers=<the_number_of_workers_for_dataloader> \
      --modelPath=<directory_for_loading_model> \
      --predict=<the_output_path_of_testing_data_prediction>
   ```

### Slot Tagging
To train and test the slot tagging model:
1. Train the model:
   ```bash
      python3 src/p2_train_tags.py \
      --train_data=<path_to_json_of_training_data> \
      --val_data=<path_to_json_file_of_validation_data> \
      --label_idx=<path_to_the_json_file_of_labelindex> \
      --word_vector=<path_to_the_word_embedding_data> \
      --vocab=<path_to_the_Class_Vocab_containing_the_token_information> \
      --sequence_len=<max_length_of_sequence_tokens> \
      --learning_rate=<learning_rate_of_optimization> \
      --batchsize=<batch_size> \
      --num_workers=<the_number_of_workers_for_dataloader> \
      --epoch=<the_number_of_epoch> \
      --model_dir=<directory_for_saving_model>

   ```
2. Test the model:
   ```bash
      python3 src/p2_test_tags.py \
      --test_data=<path_to_json_of_test_data> \
      --label_idx=<path_to_the_json_file_of_labelindex> \
      --word_vector=<path_to_the_word_embedding_data> \
      --vocab=<path_to_the_Class_Vocab_containing_the_token_information> \
      --sequence_len=<max_length_of_sequence_tokens> \
      --num_workers=<the_number_of_workers_for_dataloader> \
      --modelPath=<directory_for_loading_model> \
      --predict=<the_output_path_of_testing_data_prediction>
    
   ```

---

## Project Structure
- **`p1_model.py`**: Defines the model for intent classification.
- **`p2_model.py`**: Defines the model for slot tagging.
- **`p1_train_intent.py`**: Training script for intent classification.
- **`p2_train_tags.py`**: Training script for slot tagging.
- **`p1_utils.py` & `p2_utils.py`**: Utility functions for data processing.
- **`intent_cls.sh` & `slot_tag.sh`**: Shell scripts for managing tasks.
- **`environment.yml`**: Environment dependencies for the project.
- **`requirements.in`**: Additional Python dependencies.
- **`README.md`**: This file, providing an overview of the project.

---

## License
This project is for educational purposes as part of the Applied Deep Learning course. Please respect academic integrity when using this repository.

---

## Acknowledgments
Special thanks to the teaching team of Applied Deep Learning for providing resources and guidance.

---
