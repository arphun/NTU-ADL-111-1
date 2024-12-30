import json
import torch
from model import mc_Bert
from utils import Parse
from data import getDataLoader


def train(args, model):
    """
    Trains the multiple-choice BERT model.

    Args:
        args: Parsed command-line arguments containing training configurations.
        model (mc_Bert): The multiple-choice BERT model instance.

    """
    # Load context data
    with open(args.context, 'r') as f:
        context = json.load(f)

    # Load and prepare training data
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    train_loader = getDataLoader(
        task="MultipleChoice",
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
        task="MultipleChoice",
        mode="validating",
        tokenizer=model.tokenizer,
        context=context,
        data=val_data,
        max_q_len=args.max_q_len,
        max_p_len=args.max_p_len,
        max_s_len=args.max_s_len,
        batch_size=args.batch_size,
        window_interval=None,
        num_workers=args.num_workers,
    )
    
    model.train(train_loader, val_loader)
    return

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = mc_Bert(args=args, device=device)
    train(args, model)
    
if __name__ == "__main__":
    args = Parse()
    main(args)