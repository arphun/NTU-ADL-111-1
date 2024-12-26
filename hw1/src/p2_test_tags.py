from p2_model import slotTags
from p2_data import genDataLoader  
from p2_utils import Parse, load_model, save_csv
import torch
import pickle
import json

def test(args, device, model, vocab, tag2idx):
    """
    Perform testing on the test dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to run the model on (CPU or GPU).
        model (torch.nn.Module): Trained slot tagging model.
        vocab: Vocabulary object used for token encoding.
        tag2idx (dict): Mapping of slot tags to indices.

    Returns:
        tuple: A list of IDs and a list of predicted slot tag sequences.
    """
    print("Testing......")
    model.to(device)
    model.eval()

    # Load test data
    with open(args.test_data, 'rb') as data:
        test_data = json.load(data)

    # Create DataLoader for test data
    test_loader = genDataLoader(
        mode='testing', data=test_data, vocab=vocab, tag2idx=tag2idx, 
        sequence_len=args.sequence_len, batchsize=args.batchsize, num_workers=args.num_workers
    )
    
    id_list = []  # List to store IDs
    tags_list = []  # List to store predicted slot tags for each sequence
    tag = list(tag2idx.keys())  # List of tag names

    # Perform inference
    for _, (id, x) in enumerate(test_loader):
        # x: (batch_size, seq_len)
        x = x.to(device)
        predict = model(x)  # predict: (batch_size, seq_len, num_classes)

        # Process predictions for each sequence
        for i, seq_predict in enumerate(predict):
            _, indices = torch.max(seq_predict, dim=1)  # Predicted labels for the sequence
            beforePadLength = x[i, x[i].nonzero()].size(0)  # Length of sequence before padding
            label_list = [tag[idx] for idx in indices[:beforePadLength]]
            tags_list.append(label_list)
        id_list += list(id)

    return id_list, tags_list

def main(args):
    """
    Main function to load the model, perform testing, and save predictions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Load pre-trained word embeddings and vocabulary
    wordVector = torch.load(args.word_vector)  # wordVector.shape = [num_tokens, embedding_dim]
    with open(args.vocab, 'rb') as v:
        vocab = pickle.load(v)
    with open(args.label_idx, 'rb') as t2i:
        tag2idx = json.load(t2i)

    # Initialize the model and load pre-trained weights
    model = slotTags(128, len(tag2idx), wordVector, vocab)
    load_model(args.model_dir, args.load_model, model, None, device)

    # Perform testing
    id_list, tags_list = test(args, device, model, vocab, tag2idx)

    # Save predictions to a CSV file
    save_csv(id_list, tags_list, args.predict)

if __name__ == "__main__":
    # Parse command-line arguments and start the main function
    args = Parse()
    main(args)