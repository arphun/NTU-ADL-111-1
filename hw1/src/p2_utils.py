from argparse import ArgumentParser
import torch
import pandas as pd
import os
def print_info(eclipsed_time , ep_count, ep_total,  ep_percentage,  avg_loss, correct_percentage):
    """
    Print training progress information.

    Args:
        eclipsed_time (float): Elapsed time since the training started.
        ep_count (int): Current epoch number.
        ep_total (int): Total number of epochs.
        ep_percentage (float): Percentage of completed data in the current epoch.
        avg_loss (float): Average loss for the current epoch.
        correct_percentage (float): Accuracy percentage for the current epoch.

    Returns:
        None
    """
    print(f'[Used Time = {eclipsed_time}s], epoch[{ep_count}/{ep_total}], finish {int(ep_percentage)}% data, average loss = {avg_loss:.2f}, accuracy = {correct_percentage:.2f}')
    return
    
def save_model(dir, filename, model, optimizer):
    """
    Save the model and optimizer states to a file.

    Args:
        path (str): Path to save the model file.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.

    Returns:
        None
    """
    state = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, filename)
    torch.save(state, path)
    print(f'model saved to {path}')
    return


def load_model(dir, filename, model, optimizer, device):
    """
    Load the model and optimizer states from a file.

    Args:
        path (str): Path to the model file.
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer or None): The optimizer to load.
        device (torch.device): Device to map the model and optimizer to.

    Returns:
        None
    """
    path = os.path.join(dir, filename)
    state = torch.load(path,  map_location = device)
    model.load_state_dict(state['state_dict'])
    if(optimizer is not None):
        optimizer.load_state_dict(state['optimizer'])
    print(f'model loaded from {path}')

def save_csv(id_list, tags_list, csvpath):
    """
    Save predictions to a CSV file.

    Args:
        id_list (list): List of IDs.
        tags_list (list of lists): List of predicted slot tags for each sequence.
        csvpath (str): Path to save the CSV file.

    Returns:
        None
    """
    for i, tags in enumerate(tags_list):
         tags_list[i] = " ".join(tags)
    df = pd.DataFrame({'id' : id_list , 'tags' : tags_list})
    df.to_csv(csvpath , index = False)
    return    


def Parse():
    """
    Parse command-line arguments for training, validating, and testing configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    my_parser = ArgumentParser(description="Training slot_tags model for ADL 2022 Fall HW1 Problem2")
    #data
    my_parser.add_argument("--train_data", type=str, default="data/slot/train.json", help="Path to the json file of training data")
    my_parser.add_argument("--val_data", type=str, default="data/slot/eval.json", help="Path to the json file of validation data")
    my_parser.add_argument("--test_data", type=str, default="data/slot/test.json", help="Path to the json file of test data")
    my_parser.add_argument("--label_idx", type=str, default="cache/slot/tag2idx.json", help="Path to the json file of labelindex")
    #word embedding
    my_parser.add_argument("--word_vector", type=str, default="cache/slot/embeddings.pt", help="Path to the word embedding data")
    my_parser.add_argument("--vocab", type=str, default="cache/slot/vocab.pkl", help="Path to the Class Vocab containing the token information")
    my_parser.add_argument("--sequence_len", type=int, default="40", help="max length of sequence tokens")
    
    #hyperparameter
    my_parser.add_argument("--learning_rate", type=float, default="0.0001", help="learning rate of optimization")
    my_parser.add_argument("--batchsize", type=int, default="64", help="batch size")
    my_parser.add_argument("--num_workers", type=int, default="8", help="the number of workers for dataloader")
    my_parser.add_argument('--epoch' , type=int , default=100, help="the number of epoch")

    #model save & load
    my_parser.add_argument("--model_dir", type=str, default= "p2_slotTag_model", help="directory for saving model")
    my_parser.add_argument("--load_model", type=str, default= "slotTag_best.pth", help="filename of model")
    my_parser.add_argument("--predict", type=str, default= "predict.csv", help="the output of testing data")
    
    return my_parser.parse_args()

