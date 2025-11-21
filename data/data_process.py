import datasets
import transformers
from threading import Lock
from torch.utils.data import DataLoader
from typing import Tuple
import torch


def load_and_templated_qa_data(dataset_name: str, split: str = 'train', q_col: str = 'Question',
                               a_col: str = 'Answer') -> datasets.Dataset:
    """Load dataset and apply a conversation template to each sample."""
    dataset = datasets.load_dataset(dataset_name, split=split)

    def apply_template(samples):
        questions = samples[q_col]
        answers = samples[a_col]
        templated_samples = []
        for q, a in zip(questions, answers):
            temp = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}]
            templated_samples.append(temp)
        return {'message': templated_samples}

    dataset = dataset.map(apply_template, batched=True)
    return dataset


def split_dataset(dataset: datasets.Dataset, train_size: float = 0.8, seed: int = 42,
                  stratify_column: str = 'qtype', threshold: int = 50) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Split dataset into training and validation sets."""
    if stratify_column and stratify_column in dataset.column_names:
        # process rare classes to ensure each split has at least one sample
        counts = dataset.to_pandas()[stratify_column].value_counts()
        rare_classes = counts[counts < threshold].index.tolist()

        def merge_rare_classes(example):
            if example[stratify_column] in rare_classes:
                example[stratify_column] = 'rare_class'
            return example

        dataset = dataset.map(merge_rare_classes)
        labels = dataset.unique(stratify_column)
        class_feature = datasets.ClassLabel(num_classes=len(labels), names=labels)
        if class_feature:
            dataset = dataset.cast_column(stratify_column, class_feature)
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed, stratify_by_column=stratify_column)
    return split_datasets['train'], split_datasets['test']


class DataProcessor:
    __lock = Lock()
    __instance = None

    def __new__(cls, tokenizer_name: str):
        with cls.__lock:
            if cls.__instance is None:
                cls.__instance = super(DataProcessor, cls).__new__(cls)
                cls.__instance.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        return cls.__instance

    def __init__(self, tokenizer_name: str):
        pass  # Initialization is handled in __new__

    def tokenize(self, dataset: datasets.Dataset, message_col: str, max_length: int = 2048,
                 batch_size: int = 32) -> DataLoader:
        def tokenize_function(samples):
            messages = samples[message_col]
            batch_prompts = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for
                             msg in messages]
            batch_user_prompts = [
                self.tokenizer.apply_chat_template(msg[:-1], tokenize=False, add_generation_prompt=False) for msg in
                messages]
            tokenized = self.tokenizer(batch_prompts, padding='max_length', truncation=True,
                                       max_length=max_length, return_tensors='pt')
            tokenized_user = self.tokenizer(batch_user_prompts, padding='max_length', truncation=True,
                                            max_length=max_length, return_tensors='pt')
            labels_list = []
            full_len = tokenized['input_ids'].shape[1]
            for i in range(len(messages)):
                labels = tokenized['input_ids'][i].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
                user_len = int((tokenized_user['input_ids'][i] != self.tokenizer.pad_token_id).sum().item())
                labels[:user_len] = -100  # Mask user prompt tokens
                assert labels.shape[0] == full_len, f"labels length {labels.shape[0]} != seq_len_full {full_len}"
                labels_list.append(labels)

            tokenized['labels'] = torch.stack(labels_list, dim=0)
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
        return dataloader
