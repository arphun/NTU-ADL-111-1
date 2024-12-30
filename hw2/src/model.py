import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from time import time
from utils import print_info
# Utility function to compute cross limitations for masking
def make_cross_limitation(x):
    """
    Creates a cross-product mask to combine conditions between tokens.

    Args:
        x (torch.Tensor): A tensor of token conditions (e.g., attention mask or token types).

    Returns:
        torch.BoolTensor: A cross-product boolean mask.
    """
    return (x.unsqueeze(1) * x.unsqueeze(0)).bool()

# Utility function to compute cross limitations for masking
def cross_addition(x, y): # reference : https://dongkwan-kim.github.io/blogs/all-possible-combinations-of-row-wise-addition-using-pytorch/
    """
    Computes all possible combinations of row-wise addition between two tensors.

    Args:
        x (torch.Tensor): First tensor for row-wise addition.
        y (torch.Tensor): Second tensor for row-wise addition.

    Returns:
        torch.Tensor: Resultant tensor with all combinations.
    """
    return (x.unsqueeze(1) + y.unsqueeze(0))


# Class for managing a multiple-choice task using BERT
class mc_Bert():
    """
    A class for managing a multiple-choice task using BERT.

    Attributes:
        args: Parsed command-line arguments or configurations.
        device: Device to run the model (CPU/GPU).
        model: Pretrained BERT model for multiple-choice tasks.
        tokenizer: Tokenizer for processing input data.
        optim: Optimizer for training the model.
    """

    def __init__(self, args, device):
        """
        Initializes the multiple-choice BERT model, tokenizer, and optimizer.

        Args:
            args: Parsed command-line arguments or configurations.
            device (torch.device): Device to run the model (CPU/GPU).
        """
        self.args = args
        self.device = device

        print("Load multiple choice model : ", args.mc_model_name)
        self.model = AutoModelForMultipleChoice.from_pretrained(args.mc_model_name)
        self.model.to(self.device)
        print("Load tokenizer : ", args.mc_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.mc_model_name)
        self.optim = AdamW(self.model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay)

        return

    def set_model(self, mode, device):
        self.model.to(device)
        if(mode == "training" or mode == "train"):
            self.model.train()
        else:
            self.model.eval()

    def train(self, train_loader, val_loader):
        """
        Trains the multiple-choice model.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        """
        max_acc = 0
        save_count = 0
        start_time = time()

        # Initialize scheduler for learning rate adjustment
        total_step_num = self.args.epoch * len(train_loader) // self.args.iter_num
        scheduler = get_linear_schedule_with_warmup(
            self.optim, total_step_num // self.args.warmup_division, total_step_num
        )

        print(f"Training {self.args.epoch} epochs and each epoch contains {len(train_loader)} batches.")
        for ep_count in range(self.args.epoch):
            self.set_model("train", self.device)
            total_loss, correct_count, used_data = 0, 0, 0
            batch_total = len(train_loader)

            self.optim.zero_grad()

            for batch_count, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                # Calculate results
                loss, correct = self.compute_results(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                total_loss += loss
                correct_count += correct
                used_data += input_ids.size(0)

                # Perform optimizer step and scheduler step
                if (batch_count + 1) % self.args.iter_num == 0 or (batch_count + 1) == len(train_loader):
                    self.optim.step()
                    scheduler.step()
                    self.optim.zero_grad()

                # Print progress
                if batch_count % 50 == 0:
                    print_info(
                        eclipsed_time=time() - start_time,
                        ep_count=ep_count,
                        ep_total=self.args.epoch,
                        ep_percentage=100.0 * (batch_count / batch_total),
                        avg_loss=100.0 * (total_loss / used_data),
                        correct_percentage=100.0 * (correct_count / used_data),
                    )
            print("")
            # Evaluate model after each epoch
            acc = self.evaluate(val_loader)
            if acc >= max_acc:
                max_acc = acc
                print(f"New max accuracy: {max_acc:.2f}")
                self.save_model(self.args.model_dir, save_count, acc)
                save_count += 1

        return
    
    def compute_results(self, input_ids, attention_mask, token_type_ids, labels):
        """
        Performs forward pass and computes results.

        Args:
            input_ids (torch.Tensor): Input IDs for the batch (shape: [batch_size, seq_len]).
            attention_mask (torch.Tensor): Attention mask (shape: [batch_size, seq_len]).
            token_type_ids (torch.Tensor): Token type IDs (shape: [batch_size, seq_len]).
            labels (torch.Tensor): Ground truth labels (shape: [batch_size]).

        Returns:
            tuple: Loss value (float) and number of correct predictions (int) if labels are provided.
                   output_logits (torch.Tensor) otherwise.
        """
        
        if(labels is None):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )     
            return output.logits
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )    
            if(self.model.training):
                output.loss.backward()
            pred = output.logits.argmax(dim=1)
            loss = output.loss.item()
            correct_count = (labels == pred).sum().item()
            return loss, correct_count
    


    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

        Returns:
            float: Validation accuracy.
        """
        print("\nEvaluating :")
        self.set_model("eval", self.device)
        start_time = time()

        total_loss, correct_count, used_data = 0, 0, 0
        batch_total = len(val_loader)

        start_time = time()
        with torch.no_grad():
            for batch_count, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                loss, correct = self.compute_results(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                total_loss += loss
                correct_count += correct
                used_data += input_ids.size(0)

                if batch_count % 50 == 0:
                    print_info(
                        eclipsed_time=time() - start_time,
                        ep_count=0,
                        ep_total=0,
                        ep_percentage=100.0 * (batch_count / batch_total),
                        avg_loss=total_loss / used_data,
                        correct_percentage=100.0 * (correct_count / used_data),
                    )
        val_acc = correct_count / used_data
        print(
            f"\nEvaluation ({time() - start_time:.2f}s): loss = {total_loss / used_data:.6f}, acc = {val_acc:.6f}"
        )
        return val_acc
    
    def save_model(self, model_dir, save_count, acc):
        """
        Saves the model to the specified directory.

        Args:
            model_dir (str): Directory to save the model.
            save_count (int): Save counter for versioning.
            acc (float): Current accuracy for naming the file.
        """

        path = os.path.join(model_dir, f"mc_Bert_{save_count}_{acc:.2f}")
        self.model.save_pretrained(path)
        print(f'model saved to {path}')
        path = os.path.join(model_dir, f"mc_Bert_best")
        self.model.save_pretrained(path)
        print(f'model saved to {path}')
        return

    def load_model(self, dir):
        """
        Loads a pretrained model from the specified directory.

        Args:
            dir (str): Directory containing the saved model.
        """
        config_path = os.path.join(dir , 'config.json')
        with open(config_path , 'r') as file:
            config = json.load(file)
        
        self.model =  AutoModelForMultipleChoice.from_pretrained(dir)
        self.tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'])
        print(f'MC model loaded from {dir}')
        return 



class qa_Bert():
    """
    A class to manage Question Answering (QA) tasks using a BERT-based model.

    Attributes:
        args: Parsed command-line arguments containing configurations.
        device: The device (CPU/GPU) to run the model.
        model: Pretrained QA BERT model.
        tokenizer: Tokenizer for input processing.
        optim: Optimizer for training the model.
    """
    def __init__(self, args, device):
        """
        Initializes the QA BERT model, tokenizer, and optimizer.

        Args:
            args: Parsed command-line arguments containing configurations.
            device (torch.device): The device (CPU/GPU) to run the model.
        """
        
        self.args = args
        self.device = device
        print("Load question answering model:", args.qa_model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_name)
        self.model.to(self.device)
        print("Load tokenizer:", args.qa_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.qa_model_name)
        self.optim = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        return
    def set_model(self, mode, device):
        self.model.to(device)
        if(mode == "training" or mode == "train"):
            self.model.train()
        else:
            self.model.eval()

    def train(self, train_loader, val_loader):
        """
        Trains the QA model.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        """
        max_acc = 0
        save_count = 0
        start_time = time()
        
        # Scheduler for learning rate adjustment
        total_step_num = self.args.epoch * len(train_loader) // self.args.iter_num
        scheduler = get_linear_schedule_with_warmup(
            self.optim, total_step_num // self.args.warmup_division, total_step_num
        )
        
        # Track metrics across epochs
        train_acc_list, val_acc_list = [], []
        train_loss_list, val_loss_list = [], []

        for ep_count in range(self.args.epoch):
            self.set_model("train", self.device)
            total_loss, correct_count, used_data = 0, 0, 0
            batch_total = len(train_loader)

            self.optim.zero_grad()

            for batch_count, (input_ids, attention_mask, token_type_ids, ans) in enumerate(
                train_loader
            ):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                ans['start'] = ans['start'].to(self.device)
                ans['end'] = ans['end'].to(self.device)

                loss, correct = self.compute_results(
                    input_ids_batch=input_ids,
                    attention_mask_batch=attention_mask,
                    token_type_ids_batch=token_type_ids,
                    ans_batch=ans,
                )

                total_loss += loss
                correct_count += correct
                used_data += input_ids.size(0)

                if (batch_count + 1) % self.args.iter_num == 0 or (batch_count + 1) == len(train_loader):
                    self.optim.step()
                    scheduler.step()
                    self.optim.zero_grad()

                if batch_count % 50 == 0:
                    print_info(
                        eclipsed_time=time() - start_time,
                        ep_count=ep_count,
                        ep_total=self.args.epoch,
                        ep_percentage=100.0 * (batch_count / batch_total),
                        avg_loss=total_loss / used_data,
                        correct_percentage=100.0 * (correct_count / used_data),
                    )
            print("")
            # Evaluate after each epoch
            val_acc, val_loss = self.evaluate(val_loader)

            # Track metrics
            train_acc_list.append(correct_count / used_data)
            train_loss_list.append(total_loss / used_data)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

            # Save model if performance improves
            if val_acc >= max_acc:
                max_acc = val_acc
                print(f"New maximum accuracy: {max_acc:.2f}")
                self.save_model(self.args.model_dir, save_count)
                save_count += 1

        return
    
    def compute_results(self, input_ids_batch, attention_mask_batch, token_type_ids_batch, ans_batch):
        """
        Performs forward pass and computes results.

        Args:
            input_ids_batch (torch.Tensor): Tokenized input IDs (shape: [batch_size, seq_len]).
            attention_mask_batch (torch.Tensor): Attention masks for inputs (shape: [batch_size, seq_len]).
            token_type_ids_batch (torch.Tensor): Token type IDs to distinguish segments.
            ans_batch (dict): Dictionary containing the ground truth answers with 'start' and 'end' positions.

        Returns:
            tuple: Loss value (float) and number of correct predictions (int) if labels are provided.
                   output (torch.Tensor) otherwise.
        """  
        if(ans_batch is None):
            output = self.model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                token_type_ids=token_type_ids_batch,
            )   
            return output
        else:
            output = self.model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                token_type_ids=token_type_ids_batch,
                start_positions=ans_batch['start'],
                end_positions=ans_batch['end']
            )   
            if(self.model.training):
                output.loss.backward()
            prediction = self.calculate_prediction(
                p_start=output.start_logits, 
                p_end=output.end_logits, 
                input_ids_batch=input_ids_batch, 
                attention_mask_batch=attention_mask_batch, 
                token_type_ids_batch=token_type_ids_batch)
            loss = output.loss.item()
            correct_count = torch.tensor(ans_batch['text'] == prediction).sum().item()
            return loss, correct_count
    
    def calculate_prediction(self, p_start, p_end, input_ids_batch, attention_mask_batch, token_type_ids_batch):
        """
        Calculates predictions for answer spans in the Question Answering (QA) task.

        Args:
            p_start (torch.Tensor): Logits for the start positions of the answer (shape: [batch_size, seq_len]).
            p_end (torch.Tensor): Logits for the end positions of the answer (shape: [batch_size, seq_len]).
            input_ids_batch (torch.Tensor): Tokenized input IDs (shape: [batch_size, seq_len]).
            attention_mask_batch (torch.Tensor): Attention masks for inputs (1 for valid tokens, 0 for padding).
            token_type_ids_batch (torch.Tensor): Token type IDs to distinguish question and paragraph segments.
            para (str or None): Paragraph text (used in validation/testing mode).

        Returns:
            list or str: Predicted answer spans. 
                - In training: List of predicted answer spans for each example in the batch.
                - In validation/testing: The most probable answer span (str).
        """
        # Define negative infinity for masking invalid logits
        neg_inf = (-1.0 / torch.zeros(p_start[0].size())).to(self.device)

        if(self.model.training):
            # Training mode: Generate predictions for each example in the batch
            predictions = []

            for i in range(p_start.size(0)):
                # Compute pairwise logits for start and end positions
                logits = cross_addition(p_start[i], p_end[i])  # Shape: [seq_len, seq_len]
                # logits[x, y] represents the probability of start=x, end=y.

                # Mask invalid positions
                no_padding = make_cross_limitation(attention_mask_batch[i])  # Exclude padding tokens
                para_sep_pos = torch.argmin(attention_mask_batch[i]) - 1  # Locate separator token
                token_type_ids_batch[i, para_sep_pos] = 0  # Temporarily mask separator token
                in_paragraph = make_cross_limitation(token_type_ids_batch[i])  # Exclude non-paragraph tokens
                valid_order = torch.triu(logits) != 0  # Ensure start <= end
                
                # Apply masking
                logits = torch.where(no_padding * in_paragraph * valid_order, logits, neg_inf)
                
                # Identify the best start and end positions
                flat_index = torch.argmax(logits)
                start_tok_pos, end_tok_pos = divmod(flat_index.item(), logits.size(1))

                # Decode the predicted text
                predicted_text = self.tokenizer.decode(input_ids_batch[i, start_tok_pos: end_tok_pos + 1])
                predictions.append("".join(predicted_text.split()))  # Remove spaces

            return predictions
        
        else: # validating and testing , batch_size is always 1, the first dimension = # of choices
            # Validation/Testing mode: Process paragraph chunks to find the best answer
            ######para_enc = self.tokenizer(para, truncation=False, add_special_tokens=False)  # Tokenize paragraph
            max_probability = -torch.inf  # Track the highest probability
            best_answer = None  # Store the final answer
            
            for chunk_idx in range(p_start.size(0)):  # Iterate over paragraph chunks
                # Compute pairwise logits for start and end positions
                logits = cross_addition(p_start[chunk_idx], p_end[chunk_idx]).to(self.device)  # Shape: [seq_len, seq_len]
                # logits[x, y] represents the probability of start=x, end=y.

                # Mask invalid positions
                no_padding = make_cross_limitation(attention_mask_batch[chunk_idx]).to(self.device) # Exclude padding
                para_sep_pos = torch.argmin(attention_mask_batch[chunk_idx]) - 1  # Find separator token
                token_type_ids_batch[chunk_idx, para_sep_pos] = 0  # Mask separator token
                in_paragraph = make_cross_limitation(token_type_ids_batch[chunk_idx]).to(self.device)  # Exclude non-paragraph tokens
                valid_order = (torch.triu(logits) != 0).to(self.device)  # Ensure end >= start (upper triangular mask)

                # Apply masking
                logits = torch.where(no_padding * in_paragraph * valid_order, logits, neg_inf)
                
                # Identify the best start and end positions
                flat_index = torch.argmax(logits)
                start_tok_pos, end_tok_pos = divmod(flat_index.item(), logits.size(1))
                probability = logits[start_tok_pos, end_tok_pos]
                candidate_answer = self.tokenizer.decode(input_ids_batch[chunk_idx, start_tok_pos: end_tok_pos + 1])
                # Update best answer if probability is higher
                if probability >= max_probability:
                    max_probability = probability
                    best_answer = candidate_answer
            return best_answer

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

        Returns:
            tuple: Validation accuracy (float) and validation loss (float).
        """
        print("\nEvaluating :")
        self.set_model("eval", self.device)
        start_time = time()

        total_loss, correct_count, used_data = 0, 0, 0
        batch_total = len(val_loader)


        with torch.no_grad():
            for batch_count, (input_ids , attention_mask , token_type_ids, ans) in enumerate(val_loader):
                input_ids = input_ids.view((-1, self.args.max_s_len)).to(self.device)
                token_type_ids = token_type_ids.view((-1, self.args.max_s_len)).to(self.device)
                attention_mask = attention_mask.view((-1, self.args.max_s_len)).to(self.device)
                ans['start'] = ans['start'].view((-1)).to(self.device)
                ans['end'] = ans['end'].view((-1)).to(self.device)
                ans['text'] = ans['text'][0]

                # Perform forward pass
                
                loss, correct = self.compute_results(
                    input_ids_batch=input_ids,
                    attention_mask_batch=attention_mask,
                    token_type_ids_batch=token_type_ids,
                    ans_batch=ans,
                )                
                total_loss += loss
                correct_count += correct
                used_data += 1

                if batch_count % 50 == 0:
                    print_info(
                        eclipsed_time=time() - start_time,
                        ep_count=1,
                        ep_total=1,
                        ep_percentage=100.0 * (batch_count / batch_total),
                        avg_loss=total_loss / used_data,
                        correct_percentage=100.0 * (correct_count / used_data),
                    )
                    
        val_acc = correct_count / used_data
        print(f"\nValidation Loss: {total_loss / used_data:.6f}, Accuracy: {val_acc:.6f}")
        return val_acc, total_loss / used_data
    
    def save_model(self, model_dir, save_count):
        """
        Saves the model to a specified directory.

        Args:
            model_dir (str): Path to the model directory.
            save_count (int): Count to distinguish multiple saves.
        """
        path = os.path.join(model_dir, f"qa_bert_{save_count}")
        self.model.save_pretrained(path)
        print(f'model saved to {path}')
        path = os.path.join(model_dir, f"qa_Bert_best")
        self.model.save_pretrained(path)
        print(f'model saved to {path}')

    def load_model(self, dir):
        """
        Loads a pretrained model from a specified directory.

        Args:
            dir (str): Path to the model directory.
        """
        config_path = os.path.join(dir, "config.json")
        with open(config_path, "r") as file:
            config = json.load(file)

        self.model = AutoModelForQuestionAnswering.from_pretrained(dir)
        self.tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"])
        print(f'QA model loaded from {dir}')