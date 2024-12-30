from torch.utils.data import Dataset, DataLoader
import random
import torch
class Multiple_choices_dataset(Dataset):
    """
    Dataset class for handling multiple-choice tasks.

    Args:
        mode (str): Mode of operation ('training', 'validating', 'testing').
        tokenizer: Tokenizer for encoding text.
        context (list of str): List of context paragraphs.
        data (list of dict): Dataset containing questions and associated paragraphs.
        max_q_len (int): Maximum length for the question sequence.
        max_p_len (int): Maximum length for the paragraph sequence.
        max_s_len (int): Maximum sequence length after concatenation.

    """
    def __init__(self, mode, tokenizer, context, data, max_q_len, max_p_len, max_s_len):
        self.mode = mode
        self.tokenizer = tokenizer
        self.data = data
        self.context = context
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.max_s_len = max_s_len # should be larger than or equal to max_q_len + max_p_len + 3.

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Encoded inputs and (optionally) the answer index for training and validation.
        """
        data = self.data[index]
        question = self.tokenizer(data['question'] , truncation=True, max_length=self.max_q_len, add_special_tokens = True)
        
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        for count, p_idx in enumerate(data['paragraphs']):
            para = self.tokenizer(self.context[p_idx] , truncation = True, max_length = self.max_p_len , add_special_tokens = False)
            para["input_ids"] += [self.tokenizer.sep_token_id]
            para["attention_mask"] += [1]
            para["token_type_ids"] = [1] * len(para["input_ids"])

            input_ids = question["input_ids"] + para["input_ids"]
            pad_ids = [self.tokenizer.pad_token_id] * (max(0, self.max_s_len - len(input_ids)))
            input_ids +=  pad_ids

            # generate attention_mask : 1 for input, 0 for pad
            attention_mask = question["attention_mask"] + para["attention_mask"] + [0] * len(pad_ids)
            # generate token_type_ids : 0 for question, 1 for paragraphs
            token_type_ids = question["token_type_ids"] + para["token_type_ids"] + [1] * len(pad_ids)
            if(self.mode != "testing" and data['relevant'] == p_idx):
                answer = count 

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
        
        input_ids_list = torch.tensor(input_ids_list)
        attention_mask_list = torch.tensor(attention_mask_list)
        token_type_ids_list = torch.tensor(token_type_ids_list)

        if(self.mode != "testing"):
            return input_ids_list, attention_mask_list, token_type_ids_list, answer
        else:
            return input_ids_list, attention_mask_list, token_type_ids_list, -1
    
    def __len__(self):
        return len(self.data)

class Question_Answering_dataset(Dataset):
    """
    Dataset class for handling question answering tasks.

    Args:
        mode (str): Mode of operation ('training', 'validating', 'testing').
        tokenizer: Tokenizer for encoding text.
        context (list of str): List of context paragraphs.
        data (list of dict): Dataset containing questions and answers.
        max_q_len (int): Maximum length for the question sequence.
        max_p_len (int): Maximum length for the paragraph sequence.
        max_s_len (int): Maximum sequence length after concatenation.
        window_interval (int): Sliding window interval for long paragraphs.

    """
    def __init__(self, mode, tokenizer, context, data, max_q_len, max_p_len, max_s_len, window_interval):
        self.mode = mode
        self.tokenizer = tokenizer
        self.data = data
        self.context = context
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.max_s_len = max_s_len # should be larger than or equal to max_q_len + max_p_len + 3
        self.window_interval = window_interval
        self.compare = None
        return
    
    def generate_inputs(self, q_ids, p_ids, p_start_pos, p_end_pos, ans_start, ans_end):
        #p_end_pos not included
        # Generate input_ids : [cls] + question + [sep] + parapraphs + [sep] + [pad]
        
        q_ids = [self.tokenizer.cls_token_id] + q_ids + [self.tokenizer.sep_token_id]
        p_ids = p_ids[p_start_pos : p_end_pos]
        p_ids += [self.tokenizer.sep_token_id]
        valid_len = len(q_ids + p_ids)
        pad_ids = [self.tokenizer.pad_token_id] * (max(0, self.max_s_len - valid_len))
        input_ids =  torch.tensor(q_ids + p_ids + pad_ids)

        # Generate attention_mask, 1 for input, 0 for pad
        attention_mask = torch.tensor([1] * valid_len + [0] * len(pad_ids))

        # Generate token_type_ids , 0 for question, 1 for paragraphs
        token_type_ids = torch.tensor([0] * len(q_ids) + [1] * len(p_ids + pad_ids))
        if(ans_start is None):
            return input_ids, attention_mask, token_type_ids, None, None
        else:
            # Update answer position because of concatenation
            if(p_start_pos <= ans_start and ans_start < p_end_pos):
                ans_start = len(q_ids) + (ans_start - p_start_pos)
            else:
                ans_start = self.max_s_len

            if(p_start_pos <= ans_end and ans_end < p_end_pos):
                ans_end = len(q_ids) + (ans_end - p_start_pos)
            else:
                ans_end = self.max_s_len
            return input_ids, attention_mask, token_type_ids, ans_start, ans_end

    def __getitem__(self, index):    
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Encoded inputs, attention masks, token types, and (optionally) answer positions.
        """    

        data = self.data[index]
        question_enc = self.tokenizer(data['question'] , truncation=True, max_length=self.max_q_len, add_special_tokens = False)

        para = self.context[data['relevant']]
        para_enc = self.tokenizer(para, add_special_tokens = False)

        if(self.mode != "testing"):
            answer_char_start = data['answer']['start']
            answerText = data['answer']['text']
            answer_char_end = answer_char_start + len(answerText) - 1
            ans_tok_start = para_enc.char_to_token(answer_char_start)
            ans_tok_end = para_enc.char_to_token(answer_char_end)
        else:
            ans_tok_start = None
            ans_tok_end = None

        if(self.mode == "training"):
            # Generate random paragraph containing the answer.
            para_start_pos = random.randint(max(0 , ans_tok_end - self.max_p_len + 1) , ans_tok_start)
            para_end_pos = min(para_start_pos + self.max_p_len, len(para_enc.input_ids))

            input_ids, attention_mask, token_type_ids, ans_tok_start, ans_tok_end = self.generate_inputs(question_enc["input_ids"], para_enc['input_ids'], para_start_pos, para_end_pos, ans_tok_start, ans_tok_end)

            return input_ids, attention_mask, token_type_ids, {'text' : answerText, 'start' : ans_tok_start, 'end' : ans_tok_end}
        else:
            inputs_ids_candidate = []
            attention_mask_candidate = []
            token_type_ids_candidate = []
            ans_tok_start_candidate = []
            ans_tok_end_candidate = []
            for para_start_pos in range(0, len(para_enc['input_ids']), self.window_interval):
                para_end_pos = min(para_start_pos + self.max_p_len, len(para_enc.input_ids))

                input_ids, attention_mask, token_type_ids, updated_ans_tok_start, updated_ans_tok_end = self.generate_inputs(question_enc["input_ids"], para_enc['input_ids'], para_start_pos, para_end_pos, ans_tok_start, ans_tok_end)
                inputs_ids_candidate.append(input_ids)
                attention_mask_candidate.append(attention_mask)
                token_type_ids_candidate.append(token_type_ids)

                if(self.mode == "validating"):
                    ans_tok_start_candidate.append(updated_ans_tok_start)
                    ans_tok_end_candidate.append(updated_ans_tok_end)

            inputs_ids_candidate = torch.stack(inputs_ids_candidate)
            attention_mask_candidate = torch.stack(attention_mask_candidate)
            token_type_ids_candidate = torch.stack(token_type_ids_candidate)            

            if(self.mode == "validating"):
                ans_tok_start_candidate = torch.tensor(ans_tok_start_candidate)
                ans_tok_end_candidate = torch.tensor(ans_tok_end_candidate)
                return inputs_ids_candidate , attention_mask_candidate , token_type_ids_candidate, {'text' : answerText, 'start' : ans_tok_start_candidate, 'end' : ans_tok_end_candidate}
            elif(self.mode == "testing"):
                return inputs_ids_candidate , attention_mask_candidate , token_type_ids_candidate, 0

    def __len__(self):
        return len(self.data)

def getDataLoader(task, mode, tokenizer, context, data, max_q_len, max_p_len, max_s_len, batch_size, window_interval, num_workers):
    """
    Create a DataLoader for the specified task.

    Args:
        task (str): Task type ('MultipleChoice' or 'QuestionAnswering').
        mode (str): Mode of operation ('training', 'validating', 'testing').
        tokenizer: Tokenizer for encoding text.
        context (list of str): List of context paragraphs.
        data (list of dict): Dataset containing questions and answers.
        max_q_len (int): Maximum length for the question sequence.
        max_p_len (int): Maximum length for the paragraph sequence.
        max_s_len (int): Maximum sequence length after concatenation.
        batch_size (int): Batch size.
        window_interval (int): Sliding window interval for long paragraphs.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    if(task == "MultipleChoice"):
        return DataLoader(Multiple_choices_dataset(mode, tokenizer, context, data, max_q_len, max_p_len, max_s_len), batch_size=batch_size, shuffle=(mode == "training"), num_workers= num_workers)
    elif(task == "QuestionAnswering"):
        return DataLoader(Question_Answering_dataset(mode, tokenizer, context, data, max_q_len, max_p_len, max_s_len, window_interval), batch_size=batch_size, shuffle=(mode == "training"), num_workers= num_workers)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Dummy tokenizer (replace 'bert-base-uncased' with your model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

    # Dummy context and data for testing
    context = [
        "這是第一段上下文。",
        "這是第二段上下文，包含更多資訊。",
        "這是第三段上下文，非常詳细。"
    ]

    multiple_choice_data = [
        {
            "question": "第一段上下文是什麼内容?",
            "paragraphs": [0, 1],
            "relevant": 0
        },
        {
            "question": "第二段上下文的内容是什麼？",
            "paragraphs": [1, 2],
            "relevant": 1
        }
    ]

    question_answering_data = [
        {
            "question": "非常詳細的段落是什麼？",
            "relevant": 2,
            "answer": {"start": 2, "text": "第三段"}
        },
        {
            "question": "哪裡有更多資訊？",
            "relevant": 1,
            "answer": {"start": 2, "text": "第二段"}
        }
    ]

    max_q_len = 10
    max_p_len = 10
    max_s_len = 30
    batch_size = 1
    window_interval = 2
    modes = ['training', 'validating', 'testing']
    
    for mode in modes:
        print(f"\nmode = {mode}")
        print("\nTesting Multiple_choices_dataset...")
        mc_dataset = Multiple_choices_dataset(
            mode=mode,
            tokenizer=tokenizer,
            context=context,
            data=multiple_choice_data,
            max_q_len=max_q_len,
            max_p_len=max_p_len,
            max_s_len=max_s_len
        )
        print("data at idx 0 = ", mc_dataset[0])

        print("\nTesting MultipleChoice DataLoader : ")
        
        mc_loader = getDataLoader(
            task="MultipleChoice",
            mode=mode,
            tokenizer=tokenizer,
            context=context,
            data=multiple_choice_data,
            max_q_len=max_q_len,
            max_p_len=max_p_len,
            max_s_len=max_s_len,
            batch_size=batch_size,
            window_interval=window_interval,
            num_workers=0
        )
        for batch in mc_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            print("Multiple Choice Batch Shapes:")
            print(f"Input IDs: {input_ids.shape}")
            print(f"Attention Mask: {attention_mask.shape}")
            print(f"Token Type IDs: {token_type_ids.shape}")
            print(f"Labels: {labels}")

    for mode in modes:
        print(f"\nmode = {mode}")
        print("\nTesting Question_Answering_dataset...")
        qa_dataset = Question_Answering_dataset(
            mode=mode,
            tokenizer=tokenizer,
            context=context,
            data=question_answering_data,
            max_q_len=max_q_len,
            max_p_len=max_p_len,
            max_s_len=max_s_len,
            window_interval=window_interval
        )

        qa_loader = getDataLoader(
            task="QuestionAnswering",
            mode=mode,
            tokenizer=tokenizer,
            context=context,
            data=question_answering_data,
            max_q_len=max_q_len,
            max_p_len=max_p_len,
            max_s_len=max_s_len,
            batch_size=batch_size,
            window_interval=window_interval,
            num_workers=0
        )

        for batch in qa_loader:
            input_ids, attention_mask, token_type_ids, answer_dict = batch

            print("Question Answering Batch Shapes:")
            print(f"Input IDs: {input_ids.shape}")
            print(f"Attention Mask: {attention_mask.shape}")
            print(f"Token Type IDs: {token_type_ids.shape}")
            print(f"Answer Dict: {answer_dict}")
    