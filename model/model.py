import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
import logging
from pathlib import Path
import datasets


class LoRAFTModel(nn.Module):

    def __init__(self, pretrained_model_name_or_path, lora_config: LoraConfig = None,
                 quantization_config: BitsAndBytesConfig = None, lora_dir: str = None, **kwargs):

        super(LoRAFTModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Load the base model with quantization if provided
        if quantization_config:
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                quantization_config=quantization_config,
                device_map='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map='cuda' if torch.cuda.is_available() else 'cpu'
            )
        base_model = prepare_model_for_kbit_training(base_model)

        if lora_dir is not None:
            # Load existing LoRA model
            p = Path(lora_dir)
            if not p.exists() or not p.is_dir():
                raise ValueError(f"LoRA directory {lora_dir} does not exist or is not a directory.")

            if not (p / "adapter_config.json").exists():
                raise ValueError(f"LoRA directory {lora_dir} does not contain adapter_config.json.")

            self.model = PeftModel.from_pretrained(base_model, lora_dir, **kwargs)
        else:
            if lora_config is None:
                raise ValueError("lora_config must be provided when lora_dir is not specified.")
            # Initialize LoRA
            self.model = get_peft_model(base_model, lora_config)

        trainable_params_number, all_params_number = self.model.get_nb_trainable_parameters()
        logging.info(
            f"LoRA model initialized. Trainable parameters: {trainable_params_number}, "
            f"All parameters: {all_params_number}, "
            f"trainable ratio: {trainable_params_number / all_params_number:.2%}",
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    def get_model(self):
        return self.model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, lora_config: LoraConfig = None,
                        quantization_config: BitsAndBytesConfig = None, lora_dir: str = None, **kwargs):
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path,
                   lora_config=lora_config,
                   quantization_config=quantization_config,
                   lora_dir=lora_dir,
                   **kwargs)

    def tokenize(self, dataset: datasets.Dataset, message_col: str, max_length: int = 2048) -> datasets.Dataset:
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
        return tokenized_dataset

    def re_tokenize(self, batch_idx: torch.Tensor) -> list:
        if isinstance(batch_idx, torch.Tensor):
            batch_idx = batch_idx.tolist()
        return self.tokenizer.batch_decode(batch_idx, skip_special_tokens=True)
