from p1_model import intentCls
from p1_data import genDataLoader  
from p1_utils import Parse, load_model, save_csv
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import json
import os
def test(args, device, model, vocab):
    """
    Perform testing on the test dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run the model on (CPU or GPU).
        model (nn.Module): The trained model to evaluate.
        vocab: Vocabulary object used for token encoding.

    Returns:
        tuple: A list of IDs and a list of predicted labels.
    """

    print("Testing......")
    model.to(device)
    model.eval()
    
    with open(args.label_idx, 'rb') as i2i:
        intent2idx = json.load(i2i)
    with open(args.test_data, 'rb') as data:
        test_data = json.load(data)

    test_loader = genDataLoader(mode='testing', data=test_data, vocab=vocab, intent2idx=intent2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    
    
    id_list = []
    label_list = []
    for batch_idx, (id, x) in enumerate(test_loader):
        x = x.to(device)
        predict = model(x)

        id_list += list(id)

        _, indices = torch.max(predict, dim=1)
        label_list += list(indices.tolist())

    intent = list(intent2idx.keys())
    label_list = [intent[list(intent2idx.values()).index(label)] for label in label_list]
    return id_list, label_list



            
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 單GPU或者CPU
    
    wordVector = torch.load(args.word_vector) # wordVector.shape = [token.nums, feature_length]
    with open(args.vocab, 'rb') as v:
        vocab = pickle.load(v)
    model = intentCls(150, wordVector, vocab)
    load_model(args.model_dir, args.load_model, model, None, device)

    id_list, label_list = test(args, device, model, vocab)
    
    # Save predictions to a CSV file
    save_csv(id_list, label_list, args.predict)
if __name__ == "__main__":
    args = Parse()
    main(args)