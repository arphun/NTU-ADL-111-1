from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import nltk
nltk.download('punkt')
class intentDataSet(Dataset):
    def __init__(self, mode, datas, vocab, intent2idx, sequence_len):
        """
        Custom PyTorch Dataset for handling intent classification data.

        Attributes:
            mode (str): Mode of operation ('training', 'validating', 'testing').
            data (list): Encoded sequences of token IDs.
            label (list): Encoded intent labels (only in non-testing modes).
            id (list): Unique IDs for each data point.

        Args:
            mode (str): Mode of operation ('training', 'validating', 'testing').
            datas (list of dict): Dataset containing 'text', 'intent', and 'id'.
            vocab: Vocabulary object with `token_to_id` and `pad_id` attributes.
            intent2idx (dict): Mapping of intent labels to indices.
            sequence_len (int): Maximum sequence length for padding/cutting.
        """
        self.mode = mode
        self.data = []
        self.label = []
        self.id = []
        #tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        for data in datas:
            #encoding text to sequence vector
            encoded_data = [vocab.token_to_id(token) for token in nltk.word_tokenize(data['text'])]
            self.data.append(encoded_data)

            #convert intent to y label
            if(mode != "testing"):
                self.label.append(intent2idx[data['intent']])

            self.id.append(data['id'])

        for i in range(len(self.data)):
            #padding and cut sequence
            cut_length = min(len(self.data[i]), sequence_len)
            self.data[i] = self.data[i][:cut_length]
            self.data[i] += [vocab.pad_id] * (sequence_len - cut_length)

        return
            


    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple containing ID, input tensor, and label tensor (if not in testing mode).
        """
        if(self.mode != 'testing'):
            return self.id[idx], torch.tensor(self.data[idx]), torch.tensor(self.label[idx])
        else:
            return self.id[idx], torch.tensor(self.data[idx])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

def genDataLoader(mode, data, vocab, intent2idx, sequence_len, batchsize, num_workers):
    """
    Generates a DataLoader for the intent dataset.

    Args:
        mode (str): Mode of operation ('training', 'validation', 'testing').
        data (list of dict): Dataset containing 'text', 'intent', and 'id'.
        vocab: Vocabulary object with `token_to_id` and `pad_id` attributes.
        intent2idx (dict): Mapping of intent labels to indices.
        sequence_len (int): Maximum sequence length for padding/cutting.
        batchsize (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    
    return DataLoader(intentDataSet(mode, data, vocab, intent2idx, sequence_len), batch_size=batchsize, shuffle=(mode == "training"), num_workers= num_workers)
    

#testing the correctness
if __name__ == "__main__":
    # Sample data
    data = [
        {"text": "hello world", "intent": "greeting", "id": "1"},
        {"text": "how are you", "intent": "question", "id": "2"},
        {"text": "hello", "intent": "greeting", "id": "3"}
    ]

    intent2idx = {"greeting": 0, "question": 1}

    # Dummy data for testing
    class DummyVocab:
        def __init__(self):
            self.token_to_id_map = {"hello": 1, "world": 2, "how": 3, "are": 4, "you": 5}
            self.pad_id = 0
            self.unk_id = -1 #unknown tokens

        def token_to_id(self, token):
            return self.token_to_id_map.get(token, self.unk_id)  # 0 for unknown tokens

    vocab = DummyVocab()

    sequence_len = 5
    batchsize = 2
    num_workers = 0

    # Testing Dataset
    modes = ["training", "validating", "testing"]
    for mode in modes: 
        print("Test {} dataset".format(mode))
        dataset = intentDataSet(mode, data, vocab, intent2idx, sequence_len)
        print("Dataset size:", len(dataset))
        for i in range(len(dataset)):
            print("Sample:", dataset[i])

    # Testing DataLoader
    for mode in modes: 
        print("\nTest {} dataloader".format(mode))
        dataloader = genDataLoader(mode, data, vocab, intent2idx, sequence_len, batchsize, num_workers)
        print("\nBatches:")
        for batch in dataloader:
            print(batch)