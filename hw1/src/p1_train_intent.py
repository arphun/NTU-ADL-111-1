from p1_model import intentCls
from p1_data import genDataLoader  
from p1_utils import Parse, print_info, save_model
from time import time
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import json
import os

def validation(args, device, model, val_loader):
    """
    Perform validation on the validation dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run the model on (CPU or GPU).
        model (nn.Module): The model being trained.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        float: Validation accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()  
    total_loss = 0
    correct_count = 0
    total_num = 0
    with torch.no_grad():
        for id, x, y in val_loader:
            x, y = x.to(device), y.to(device)
            predict = model(x)
            loss = criterion(predict, y)
            total_loss += loss.item()

            #For print_info
            _, indices = torch.max(predict , dim = 1)
            correct_count += (indices == y).sum().item()
            total_num += x.size()[0]
    print(f"Validation loss = {total_loss / total_num :.5f}, validation accuracy = { 100*(correct_count / total_num) : .2f}%")

    return correct_count / total_num

def train(args, device, model, vocab):
    """
    Train the intent classification model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run the model on (CPU or GPU).
        model (nn.Module): The model to train.
        vocab: Vocabulary object used for token encoding.

    Returns:
        None
    """
    
    print("Training......")

    # Load training and validation data
    with open(args.train_data, 'rb') as data:
        train_data = json.load(data)
    with open(args.label_idx, 'rb') as i2i:
        intent2idx = json.load(i2i)
    with open(args.val_data, 'rb') as data:
        val_data = json.load(data)

    # Initialize DataLoaders
    training_loader = genDataLoader(mode='training', data=train_data, vocab=vocab, intent2idx=intent2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    

    val_loader = genDataLoader(mode='validating', data=val_data, vocab=vocab, intent2idx=intent2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    

    # Initialize optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0
    start_time = time()
    for ep_count in range(args.epoch):
        model.train()  
        total_loss = 0
        correct_count = 0
        usedData_size = 0
        print_count = 0
        
        for batch_idx, (id, x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predict = model(x)
            loss = criterion(predict, y)
            total_loss += loss.item()

            # Calculate training accuracy
            _, indices = torch.max(predict, dim=1)
            correct_count += ((indices == y).sum()).item()
            usedData_size += x.size()[0]
            print_count += x.size()[0]

            loss.backward()
            optimizer.step()

            # Print progress every 20% of the training data
            end_time = time()
            if(print_count >  (len(train_data)/5)):
                print_count = 0
                print_info(int(end_time-start_time), ep_count, args.epoch,  100*(usedData_size/ len(train_data)),  total_loss / usedData_size, correct_count/usedData_size)
        
        # Perform validation at the end of each epoch
        accuracy = validation(args, device, model, val_loader)
        if(accuracy > max_acc):
            max_acc = accuracy
            print(f"Improved accuracy = {100 * max_acc:.2f}%")

            filename = f'intent_{100 * max_acc:.2f}.pth'
            save_model(args.model_dir, filename, model, optimizer)
            filename = f'intent_best.pth'
            save_model(args.model_dir, filename, model, optimizer)

def main(args):
    """
    Main function to initialize components and start training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wordVector = torch.load(args.word_vector) # Load pre-trained word embeddings, wordVector.shape = [token.nums, feature_length]
    with open(args.vocab, 'rb') as v:
        vocab = pickle.load(v)

    # Initialize the model
    model = intentCls(150, wordVector, vocab)
    model.to(device)

    # Start training
    train(args, device, model, vocab)

if __name__ == "__main__":
    # Parse command-line arguments and start main function
    args = Parse()
    main(args)