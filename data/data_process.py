import datasets
from typing import Tuple
from pathlib import Path


def apply_template(samples, q_col: str = 'Question', a_col: str = 'Answer'):
    questions = samples[q_col]
    answers = samples[a_col]
    templated_samples = []
    for q, a in zip(questions, answers):
        temp = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]
        templated_samples.append(temp)
    return {'message': templated_samples}


def load_and_template_local_test_data(file_path: str) -> datasets.Dataset:
    """Load local test data from a JSON or CSV file."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"File {file_path} does not exist or is not a file.")

    if p.suffix.lower() == '.json':
        dataset = datasets.load_dataset('json', data_files=file_path, split='train')
    elif p.suffix.lower() == '.csv':
        dataset = datasets.load_dataset('csv', data_files=file_path, split='train')
    else:
        raise ValueError("Unsupported file format. Only JSON and CSV are supported.")
    dataset = dataset.rename_column('0', 'Question')
    dataset = dataset.rename_column('1', 'Answer')
    dataset = dataset.map(apply_template, batched=True)

    return dataset


def load_and_templated_qa_data(dataset_name: str, split: str = 'train', q_col: str = 'Question',
                               a_col: str = 'Answer') -> datasets.Dataset:
    """Load dataset and apply a conversation template to each sample."""
    dataset = datasets.load_dataset(dataset_name, split=split)
    dataset = dataset.map(apply_template, fn_kwargs={'q_col': q_col, 'a_col': a_col}, batched=True)
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
