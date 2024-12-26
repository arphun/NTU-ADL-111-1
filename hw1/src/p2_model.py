import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class slotTags(nn.Module):
    def __init__(self, lstm_feature, nums_classes, wordVector, vocab):
        """
        Slot Tagging Model using embeddings, LSTM, and fully connected layers.

        Attributes:
            nums_classes (int): Number of output slot tag classes.
            embedding_layer (nn.Embedding): Embedding layer initialized with pre-trained word vectors.
            linear_1 (nn.Linear): Linear layer to project embedding dimensions.
            self_attention (nn.LSTM): Bi-directional LSTM layer for sequence modeling.
            linear_2 (nn.Sequential): Fully connected layers for classification.
        """
        super(slotTags , self).__init__()
        self.nums_classes = nums_classes

        # Initialize embedding layer with pre-trained word vectors
        init_weight = torch.FloatTensor(np.array(wordVector))
        self.embedding_layer = nn.Embedding(init_weight.size()[0], embedding_dim=init_weight.size()[1], padding_idx=vocab.pad_id)
        self.embedding_layer.weight = nn.Parameter(init_weight)
        self.embedding_layer.weight.requires_grad = True

        # Linear projection layer
        self.linear_1 = nn.Linear(init_weight.size()[1], lstm_feature)

        # Bi-directional LSTM for sequence modeling
        self.self_attention = nn.LSTM(input_size = lstm_feature , hidden_size = 2*lstm_feature , num_layers = 4 , dropout = 0.3 , batch_first = True, bidirectional = True)

        # Fully connected layers for classification
        self.linear_2 = nn.Sequential(
            nn.Linear(lstm_feature*4, lstm_feature*4 , bias = True) ,
            nn.BatchNorm1d(lstm_feature*4) , 
            nn.ReLU() ,
            nn.Dropout(0.5) , 
            nn.Linear(lstm_feature*4 , nums_classes , bias = True)
        )

    def forward(self, x):
        """
        Forward pass for the slot tagging model.

        Args:
            x (torch.Tensor): Input tensor containing tokenized sequences. Size: (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits for each slot tag in the sequence. Size: (batch_size, seq_len, nums_classes)
        """
        # Embedding layer: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        seq_len = x.size()[1]
        x = self.embedding_layer(x)

        # Linear projection: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_feature)
        x = self.linear_1(x)

        # LSTM: (batch_size, seq_len, lstm_feature) -> (batch_size, seq_len, 2 * lstm_feature)
        x, _ = self.self_attention(x)
        
        # Flatten: (batch_size, seq_len, 2 * lstm_feature) -> (batch_size * seq_len, 2 * lstm_feature)
        x = x.reshape((-1, x.size(2)))

        # Fully connected layers: (batch_size * seq_len, 2 * lstm_feature) -> (batch_size * seq_len, nums_classes)
        x = self.linear_2(x)
        
        # Reshape: (batch_size * seq_len, nums_classes) -> (batch_size, seq_len, nums_classes)
        x = x.view(-1, seq_len, self.nums_classes)

        return x

if __name__ == "__main__":
    # Dummy Vocabulary for Testing
    class DummyVocab:
        def __init__(self):
            self.pad_id = 0  # Padding index

    vocab = DummyVocab()

    # Dummy word vectors (e.g., 10 words with embedding size 300)
    wordVector = np.random.rand(10, 300)

    # Initialize the model
    lstm_feature = 128
    nums_classes = 5
    model = slotTags(lstm_feature, nums_classes, wordVector, vocab)

    # Dummy input tensor (batch_size=4, sequence_length=6)
    dummy_input = torch.randint(0, 10, (4, 6))  # Random token IDs

    # Forward pass
    output = model(dummy_input)
    print("Model Output Shape:", output.shape)  # Expected: (4, 6, nums_classes)
    #print("Model Output:", output)