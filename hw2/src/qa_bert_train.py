import json
import torch

from model import qa_Bert
from utils import Parse
from data import getDataLoader

def train(args, model):
    """
    Trains the Question Answering (QA) model.

    Args:
        args: Parsed command-line arguments containing training configurations.
        model (qa_Bert): The QA BERT model instance.

    Returns:
        None
    """
    # Load context data
    with open(args.context, 'r') as f:
        context = json.load(f)

    # Load and prepare training data
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    train_loader = getDataLoader(
        task="QuestionAnswering",
        mode="training",
        tokenizer=model.tokenizer,
        context=context,
        data=train_data,
        max_q_len=args.max_q_len,
        max_p_len=args.max_p_len,
        max_s_len=args.max_s_len,
        batch_size=args.batch_size // args.iter_num,
        window_interval=None,
        num_workers=args.num_workers,
    )
    
    # Load and prepare validation data
    with open(args.val_data, 'r') as f:
        val_data = json.load(f)
    val_loader = getDataLoader(
        task="QuestionAnswering",
        mode="validating",
        tokenizer=model.tokenizer,
        context=context,
        data=val_data,
        max_q_len=args.max_q_len,
        max_p_len=args.max_p_len,
        max_s_len=args.max_s_len,
        batch_size=1,  # Always set to 1 for validation
        window_interval=args.window_interval,
        num_workers=args.num_workers,
    )

    model.train(train_loader, val_loader)
    return

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    model = qa_Bert(args=args, device=device)
    train(args, model)
    
if __name__ == "__main__":
    args = Parse()
    main(args)