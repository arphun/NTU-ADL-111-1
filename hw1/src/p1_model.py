import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class intentCls(nn.Module):
    """
    Intent Classification Model using embedding, self-attention, and fully connected layers.

    Attributes:
        embedding_layer (nn.Embedding): Embedding layer initialized with pre-trained word vectors.
        linear_1 (nn.Linear): First linear transformation to adjust the embedding size.
        self_attention_layer (nn.TransformerEncoderLayer): Transformer encoder layer for self-attention.
        self_attention (nn.TransformerEncoder): Multi-layer transformer encoder for feature extraction.
        linear_2 (nn.Sequential): Fully connected layers for classification.
    """
    def __init__(self, num_classes, wordVector, vocab):
        """
        Initialize the intent classification model.

        Args:
            num_classes (int): Number of output classes.
            wordVector (list or np.ndarray): Pre-trained word embedding vectors.
            vocab: Vocabulary object with `pad_id` for padding index.
        """
        super(intentCls , self).__init__()
        init_weight = torch.FloatTensor(np.array(wordVector))
        self.embedding_layer = nn.Embedding(init_weight.size()[0], embedding_dim=init_weight.size()[1], padding_idx=vocab.pad_id)
        self.embedding_layer.weight = nn.Parameter(init_weight)
        self.embedding_layer.weight.requires_grad = True

        self.linear_1 = nn.Linear(init_weight.size()[1] , 128)
        self.self_attention_layer = nn.TransformerEncoderLayer(d_model = 128 , nhead = 4 , dim_feedforward = 128 , dropout = 0.5 , activation = 'relu' , batch_first = True)
        self.self_attention = nn.TransformerEncoder(self.self_attention_layer , num_layers = 2)

        self.linear_2 = nn.Sequential(
            nn.Linear(384 , 1024 , bias = True) ,
            nn.BatchNorm1d(1024) , 
            nn.ReLU() ,
            nn.Dropout(0.5) , 
            nn.Linear(1024 , num_classes , bias = True)
        )

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor containing tokenized sequences.

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.embedding_layer(x)
        x = self.linear_1(x)
        x = self.self_attention(x)
        # concat three useful feature : min response, max response, mean response
        x = torch.cat([x.min(dim = 1).values , x.max(dim = 1).values , x.mean(dim = 1)] , dim = 1)
        x = self.linear_2(x)
        return x
    
if __name__ == "__main__":
    # Dummy vocabulary with padding index
    class DummyVocab:
        def __init__(self):
            self.pad_id = 0  # Padding index

    vocab = DummyVocab()

    # Dummy word vectors (e.g., 10 words with embedding size 50)
    wordVector = np.random.rand(10, 50)

    num_classes = 3
    model = intentCls(num_classes, wordVector, vocab)

    # Dummy input tensor (batch_size=4, sequence_length=5)
    dummy_input = torch.randint(0, 10, (4, 5))  # Random token IDs

    # Forward pass
    output = model(dummy_input)
    print("Model Output Shape:", output.shape)  # Should be (4, num_classes)
    print("Model Output:", output)