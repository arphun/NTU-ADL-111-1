from argparse import ArgumentParser
import torch
import pandas as pd
def print_info(eclipsed_time , ep_count, ep_total,  ep_percentage,  avg_loss, correct_percentage):
    """
    Print training progress information in a formatted manner.

    Args:
        eclipsed_time (float): Elapsed time since training started.
        ep_count (int): Current epoch number.
        ep_total (int): Total number of epochs.
        ep_percentage (float): Percentage of data processed in the current epoch.
        avg_loss (float): Average loss for the current epoch or iteration.
        correct_percentage (float): Accuracy percentage for the current epoch or iteration.
    """
    print(f'\r[Used Time = {eclipsed_time:.2f}s]\tepoch[{ep_count}/{ep_total}]\tfinish {ep_percentage:.2f}% data\taverage loss = {avg_loss:.2f}\tacc = {correct_percentage:.2f}%', end="")
    return
    
def Parse():
    """
    Parse command-line arguments for training, validation, and testing configurations.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    my_parser = ArgumentParser(description="ADL 2022 Fall HW2")
    #data
    my_parser.add_argument("--context", type=str, default="data/context.json", help="Path to the json file of context")
    my_parser.add_argument("--train_data", type=str, default="data/train.json", help="Path to the json file of training data")
    my_parser.add_argument("--val_data", type=str, default="data/valid.json", help="Path to the json file of validation data")
    my_parser.add_argument("--test_data", type=str, default="data/test.json", help="Path to the json file of test data")

    #model type

    my_parser.add_argument("--mc_model_name", type=str, default="hfl/chinese-macbert-base", help="BERT model name")
    my_parser.add_argument("--qa_model_name", type=str, default="hfl/chinese-macbert-large", help="BERT model name")
    my_parser.add_argument("--pretrained", action='store_true', help="option for loading pretrained model")
    my_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    my_parser.set_defaults(pretrained=True)

    #hyperparameter
    my_parser.add_argument("--max_q_len", type=int, default=50, help="the max length of question")
    my_parser.add_argument("--max_p_len", type=int, default=450, help="the max length of paragraph")
    my_parser.add_argument("--max_s_len", type=int, default=512, help="the max length of sequence")

    my_parser.add_argument("--window_interval", type=int, default=50, help="size of paragraph interval")
    my_parser.add_argument("--learning_rate", type=float, default=0.00005, help="learning rate of optimization")
    my_parser.add_argument("--weight_decay", type=float, default=0.01, help="the value of weight decay")
    my_parser.add_argument("--warmup_division", type=int, default=10, help="the percentage of warmup, n means (1/n)% warmup")
    my_parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    my_parser.add_argument("--iter_num", type=int, default=2, help="the number of gradient accumulation")
    my_parser.add_argument("--num_workers", type=int, default=8, help="the number of workers for dataloader")
    my_parser.add_argument('--epoch' , type = int , default=5, help="the number of epoch")

    #model save & load
    my_parser.add_argument("--model_dir", type=str, default= "model", help="directory for saving/loading model")
    my_parser.add_argument("--predict", type=str, default= "predict.csv", help="the output of testing data")



    return my_parser.parse_args()
