from argparse import ArgumentParser
from p2_model import slotTags
from p2_data import genDataLoader  
from p2_utils import Parse, print_info, save_model
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
        model (torch.nn.Module): The model being trained.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
        float: Validation accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()  
    total_loss = 0
    correct_count = 0
    usedData_size = 0

    with torch.no_grad():
        for id, x, y in val_loader:
            x, y= x.to(device), y.to(device)
            predict = model(x)

            #Calculate correct count
            for i, seq_predict in enumerate(predict):  # For each sequence
                _, indices = torch.max(seq_predict, dim=1)  # Predicted labels
                beforePadLength = x[i, x[i].nonzero()].size()[0]
                if (indices[:beforePadLength] == y[i][:beforePadLength]).sum().item() == beforePadLength:
                    correct_count += 1

            # Flatten for loss calculation
            predict = predict.view(predict.size(0) * predict.size(1), predict.size(2))  # (batch_size * seq_len, num_classes)
            y = y.view(-1)
            loss = criterion(predict, y)
            total_loss += loss.item()

            usedData_size += x.size()[0]
    print(f"Validation loss = {total_loss / usedData_size :.5f}, validation accuracy = { 100*(correct_count / usedData_size) : .2f}%")
    return correct_count / usedData_size

def train(args, device, model, vocab, tag2idx):
    """
    Train the slot tagging model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run the model on (CPU or GPU).
        model (torch.nn.Module): The model to train.
        vocab: Vocabulary object used for token encoding.
        tag2idx (dict): Mapping of slot tags to indices.

    Returns:
        None
    """
    print("Training......")

    # Load training and validation data
    with open(args.train_data, 'rb') as data:
        train_data = json.load(data)
    with open(args.val_data, 'rb') as data:
        val_data = json.load(data)

    
    train_loader = genDataLoader(mode='training', data=train_data, vocab=vocab, tag2idx=tag2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    
    
    val_loader = genDataLoader(mode='validating', data=val_data, vocab=vocab, tag2idx=tag2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    
    '''
    total_loader = genDataLoader(mode='total', data=(train_data, val_data), vocab=vocab, tag2idx=tag2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers)    
    '''

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
             
        for batch_idx, (id, x, y) in enumerate(train_loader):
     
            x, y= x.to(device), y.to(device)
            optimizer.zero_grad()

            predict = model(x)  # predict: (batch_size, seq_len, num_classes)

            # Calculate correct count for sequences
            for i, seq_predict in enumerate(predict):
                _, indices = torch.max(seq_predict, dim=1)
                beforePadLength = x[i, x[i].nonzero()].size()[0]
                if (indices[:beforePadLength] == y[i][:beforePadLength]).sum().item() == beforePadLength:
                    correct_count += 1

            # Flatten for loss calculation
            predict = predict.view(predict.size(0) * predict.size(1), predict.size(2))  # (batch_size * seq_len, num_classes)
            y = y.view(-1) 
            loss = criterion(predict, y)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Print progress
            usedData_size += x.size(0)
            print_count += x.size(0)
            end_time = time()
            if(print_count >  (len(train_data)/5)):
                print_count = 0
                print_info(int(end_time-start_time), ep_count, args.epoch,  100*(usedData_size/ len(train_data)),  total_loss / usedData_size, correct_count/usedData_size)
        
        accuracy = validation(args, device, model, val_loader)
        if(accuracy > max_acc):
            max_acc = accuracy
            print(f"Get improved accuracy = {100*max_acc:.2f}%")
            filename = f'slotTag{100 * max_acc}.pth'
            save_model(args.model_dir, filename , model, optimizer)
            filename = f'slotTag_best.pth'
            save_model(args.model_dir, filename , model, optimizer)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 單GPU或者CPU
    
    wordVector = torch.load(args.word_vector) # wordVector.shape = [token.nums, feature_length]
    with open(args.vocab, 'rb') as v:
        vocab = pickle.load(v)
    with open(args.label_idx, 'rb') as t2i:
        tag2idx = json.load(t2i)

    model = slotTags(128, len(tag2idx), wordVector, vocab)
    model.to(device)
    train(args, device, model, vocab, tag2idx)
    



if __name__ == "__main__":
    args = Parse()
    main(args)