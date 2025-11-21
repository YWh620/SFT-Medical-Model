from model.model import LoRAFTModel
from data.data_process import load_and_templated_qa_data, split_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
import wandb
import matplotlib.pyplot as plt

dataset_name = "keivalya/MedQuad-MedicalQnADataset"
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"


class CollectMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get('logs', {})
        if 'loss' in logs:
            self.train_losses.append((state.global_step, logs['loss']))
        if 'eval_loss' in logs:
            self.eval_losses.append((state.global_step, logs['eval_loss']))


def plot_train_eval_curves(train_losses, eval_losses):

    train_steps, train_vals = zip(*train_losses)
    eval_steps, eval_vals = zip(*eval_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_vals, label='Train Loss')
    plt.plot(eval_steps, eval_vals, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("./results-qwen3-med-lora.png")


def train():
    # Initialize WandB
    wandb.init(entity="YWh620", project="lora-fine-tuning", name="QAWen3-medical-finetune")

    # Load and process dataset
    dataset = load_and_templated_qa_data(dataset_name, split="train")
    train_dataset, val_dataset = split_dataset(dataset, train_size=0.8, stratify_column="qtype")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    )

    # Initialize model
    model = LoRAFTModel.from_pretrained(pretrained_model_name_or_path=base_model_name, lora_config=lora_config,
                                        quantization=quantization_config)

    # Initialize DataProcessor
    train_dataset = model.tokenize(train_dataset, 'message')
    val_dataset = model.tokenize(val_dataset, 'message')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results-qwen3-med-lora",

        # —— Training ——
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        weight_decay=0.01,
        max_grad_norm=0.5,

        # —— Logging ——
        logging_dir="./logs-optimize-qwen3-med-lora",
        logging_steps=20,

        # —— Eval/Save ——
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # —— Mixed precision ——
        bf16=torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,

        # —— Others ——
        report_to=["wandb"]
    )

    collect_metrics_cb = CollectMetricsCallback()

    # Initialize Trainer
    trainer = Trainer(
        model=model.get_model(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[collect_metrics_cb]
    )

    # Start training
    trainer.train()

    # Save the final model
    model.save_pretrained("./final-model-qwen3-med-lora")

    # Plot training and evaluation curves
    plot_train_eval_curves(collect_metrics_cb.train_losses, collect_metrics_cb.eval_losses)


if __name__ == '__main__':
    print("Starting training...")
    train()
    print("Training completed.")