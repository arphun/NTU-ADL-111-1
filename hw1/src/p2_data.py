from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import nltk.data
#nltk.data.load('nltk/tokenizers/punkt/english.pickle')
class slogTagsDataSet(Dataset):
    """
    Custom PyTorch Dataset for handling slot tagging data.

    Attributes:
        mode (str): Mode of operation ('training', 'validation', 'testing').
        data (list): Encoded sequences of token IDs.
        tags (list): Encoded tag labels (only in non-testing modes).
        id (list): Unique IDs for each data point.

    Args:
        mode (str): Mode of operation ('training', 'validation', 'testing').
        datas (list of dict): Dataset containing 'tokens', 'tags', and 'id'.
        vocab: Vocabulary object with `encode` and `pad_id` attributes.
        tag2idx (dict): Mapping of slot tags to indices.
        sequence_len (int): Maximum sequence length for padding/cutting.
    """
    def __init__(self, mode, datas, vocab, tag2idx, sequence_len):
        self.mode = mode
        self.data = []
        self.tags = []
        self.id = []
        count = 0.

        # Process each data point in the dataset
        for data in datas:
            count += len(data['tokens'])

            # Tokenize each token and encode using the vocabulary
            tokenized = [nltk.word_tokenize(token)[0] if (len(token) >= 1) else '[PAD]' for token in data['tokens']]
            encoded_data = vocab.encode(tokenized)
            self.data.append(encoded_data)

            # Convert slot tags to indices (only for non-testing modes)
            if mode != "testing":
                self.tags.append([tag2idx[tag] for tag in data['tags']])
            self.id.append(data['id'])

        # Pad or truncate each sequence to `sequence_len`
        for i in range(len(self.data)):
            cut_length = min(len(self.data[i]), sequence_len)
            self.data[i] = self.data[i][:cut_length]
            self.data[i] += [vocab.pad_id] * (sequence_len - cut_length)
            
            if mode != "testing":
                self.tags[i] = self.tags[i][:cut_length]
                self.tags[i] += [tag2idx['Pad']] * (sequence_len - cut_length)
        return
        


    def __getitem__(self, idx):
        """
        Retrieve a data sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing ID, input tensor, and tag tensor (if not in testing mode).
        """

        if(self.mode != 'testing'):

            return self.id[idx], torch.tensor(self.data[idx]), torch.tensor(self.tags[idx])
        else:
            return self.id[idx], torch.tensor(self.data[idx])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

def genDataLoader(mode, data, vocab, tag2idx, sequence_len, batchsize, num_workers):
    """
    Generates a DataLoader for the slot tagging dataset.

    Args:
        mode (str): Mode of operation ('training', 'validation', 'testing').
        data (list of dict): Dataset containing 'tokens', 'tags', and 'id'.
        vocab: Vocabulary object with `encode` and `pad_id` attributes.
        tag2idx (dict): Mapping of slot tags to indices.
        sequence_len (int): Maximum sequence length for padding/cutting.
        batchsize (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    return DataLoader(slogTagsDataSet(mode, data, vocab, tag2idx, sequence_len), batch_size=batchsize, shuffle=(mode == "training"), num_workers= num_workers)

if __name__ == "__main__":

    data = [
        {"tokens": ["hello", "world"], "tags": ["O", "B-PER"], "id": "1"},
        {"tokens": ["how", "are", "you"], "tags": ["O", "O", "B-PER"], "id": "2"},
        {"tokens": ["hello"], "tags": ["B-PER"], "id": "3"}
    ]

    # Dummy vocabulary and tag2idx for testing
    class DummyVocab:
        def __init__(self):
            self.token_to_id = {"hello": 1, "world": 2, "how": 3, "are": 4, "you": 5, "[PAD]": 0}
            self.pad_id = 0  # Padding ID
            self.unk_id = -1
        def encode(self, tokens):
            return [self.token_to_id.get(token, self.unk_id) for token in tokens]  # 0 for unknown tokens

    vocab = DummyVocab()

    tag2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "Pad": 3}

    sequence_len = 5
    batchsize = 2
    num_workers = 0

    # Testing Dataset
    modes = ["training", "validating", "testing"]
    for mode in modes: 
        print("Test {} dataset".format(mode))
        dataset = slogTagsDataSet(mode, data, vocab, tag2idx, sequence_len)
        print("Dataset size:", len(dataset))
        for i in range(len(dataset)):
            print("Sample:", dataset[i])

     # Testing DataLoader
    for mode in modes: 
        print("\nTest {} dataloader".format(mode))
        dataloader = genDataLoader(mode, data, vocab, tag2idx, sequence_len, batchsize, num_workers)
        print("\nBatches:")
        for batch in dataloader:
            print(batch)