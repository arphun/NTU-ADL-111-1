import os
import torch
import transformers
import pandas as pd
from time import time
from utils import Parse
from model import qa_Bert, mc_Bert
from data import getDataLoader
import json

def process_mc_model_batch(model, iterator, device):
    """
    Processes a single batch for a given multiple choice model and returns the logits.

    Args:
        model: The model used for computation.
        iterator: The iterator of the DataLoader.
        device: The device (CPU or GPU).

    Returns:
        torch.Tensor: The computed logits for the batch.
    """
    input_ids, attention_mask, token_type_ids, _ = next(iterator)
    input_ids, attention_mask, token_type_ids = (
        input_ids.to(device),
        attention_mask.to(device),
        token_type_ids.to(device),
    )
    return model.compute_results(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )

def process_qa_model_batch(model, iterator, device, sequence_max_length):
    """
    Processes a single batch for a given question answering model and returns the outputs.

    Args:
        model: The model used for computation.
        iterator: The iterator of the DataLoader.
        device: The device (CPU or GPU).

    Returns:
        torch.Tensor: The computed outputs for the batch.
    """
    input_ids, attention_mask, token_type_ids, _ = next(iterator)
    input_ids = input_ids.view((-1, sequence_max_length)).to(device)
    token_type_ids = token_type_ids.view((-1, sequence_max_length)).to(device)
    attention_mask = attention_mask.view((-1, sequence_max_length)).to(device)

    return model.compute_results(
        input_ids_batch=input_ids,
        attention_mask_batch=attention_mask,
        token_type_ids_batch=token_type_ids,
        ans_batch=None,
    )


def test(args, context, test_data, mc_model, qa_model, device):
    """
    Perform testing for multiple-choice and question-answering tasks.

    Args:
        args: Parsed arguments for configurations and paths.
        context (dict): Context paragraphs mapped by their IDs.
        test_data (list): Test data containing questions and candidate paragraphs.
        mc_model (list): List of multiple-choice models for ensemble predictions.
        qa_model (list): List of question-answering models for ensemble predictions.

    Returns:
        list: Test data augmented with predicted answers.
    """

    # Multiple-choice testing
    mc_test_iters = []
    for model in mc_model:
        model.set_model("eval", device)
    
        mc_test_dataloader = getDataLoader(
            "MultipleChoice", "testing", model.tokenizer, context, test_data,
            args.max_q_len, args.max_p_len, args.max_s_len, args.batch_size, args.window_interval, args.num_workers
        )
        mc_test_iters.append(iter(mc_test_dataloader))
    
    choices = []
    with torch.no_grad():
        while True:
            try:
                logits = [process_mc_model_batch(model, iter, device) for model, iter in zip(mc_model, mc_test_iters)]
                logits = torch.stack(logits).mean(dim=0)
                choices += torch.max(logits, dim=1).indices.tolist()
            except StopIteration:
                print("Finish multiple choices.")
                break

    for i, test_item in enumerate(test_data):
        test_item['relevant'] = test_item['paragraphs'][choices[i]]

    # Question-answering testing
    print("Performing question answering...")

    qa_test_iters = []
    for model in qa_model:
        model.set_model("eval", device)
    
        qa_test_dataloader = getDataLoader(
            "QuestionAnswering", "testing", model.tokenizer, context, test_data,
            args.max_q_len, args.max_p_len, args.max_s_len, 1, args.window_interval, args.num_workers
        )
        qa_test_iters.append(iter(qa_test_dataloader))
    
    qa_main_iter = iter(
        getDataLoader(
            "QuestionAnswering", "testing", qa_model[0].tokenizer, context, test_data,
            args.max_q_len, args.max_p_len, args.max_s_len, 1, args.window_interval, args.num_workers
        )
    )

    predictions = []

    with torch.no_grad():
        while(True):
            try:
                outputs = [process_qa_model_batch(model, iter, device, args.max_s_len) for model, iter in zip(qa_model, qa_test_iters)]
                start_logits = [output.start_logits for output in outputs]
                start_logits = torch.stack(start_logits).mean(dim=0)
                end_logits = [output.end_logits for output in outputs]
                end_logits = torch.stack(end_logits).mean(dim=0)

                input_ids, attention_mask, token_type_ids, _ = next(qa_main_iter)
                input_ids = input_ids.view((-1, args.max_s_len)).to(device)
                attention_mask = attention_mask.view((-1, args.max_s_len)).to(device)
                token_type_ids = token_type_ids.view((-1, args.max_s_len)).to(device)
                prediction = qa_model[0].calculate_prediction(
                    p_start=start_logits, 
                    p_end=end_logits, 
                    input_ids_batch=input_ids, 
                    attention_mask_batch=attention_mask, 
                    token_type_ids_batch=token_type_ids
                )
                predictions += ["".join(prediction.split())]
                print(f"\r{len(predictions) + 1}/{len(qa_main_iter)}", end="")
            except StopIteration:
                print("\nFinish question answering.")
                break
    for i, prediction in enumerate(predictions):
        test_data[i]['answer'] = prediction

    return test_data


def save_prediction(data, output_file):
    """
    Save predictions to a CSV file.

    Args:
        data (list): List of test data with predicted answers.
        output_file (str): File path to save predictions.
    """
    ids, answers = [], []
    for item in data:
        ids.append(item['id'])
        answers.append(item['answer'])

    df = pd.DataFrame({'id': ids, 'answer': answers})
    df.to_csv(output_file, index=False)


def main(args):
    """
    Main function to load data, models, and perform predictions.

    Args:
        args: Parsed arguments for configurations and paths.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")
    with open(args.context, 'r') as file:
        context = json.load(file)

    with open(args.test_data, 'r') as file:
        test_data = json.load(file)

    # Load multiple-choice models
    mc_paths = ['mc_Bert_best', 'mc_Bert_0']
    mc_models = []
    for mc_model_path in mc_paths: 
        mc = mc_Bert(args, device)
        mc.load_model(os.path.join(args.model_dir, mc_model_path))
        mc_models.append(mc)
    # Load question-answering models
    qa_paths = ['qa_Bert_best', 'qa_Bert_best']
    qa_models = []
    for qa_model_path in qa_paths: 
        qa = qa_Bert(args, device)
        qa.load_model(os.path.join(args.model_dir, qa_model_path))
        qa_models.append(qa)
    # Perform testing and save predictions
    test_data = test(args, context, test_data, mc_models, qa_models, device)
    save_prediction(test_data, args.predict)


if __name__ == '__main__':
    args = Parse()
    main(args)