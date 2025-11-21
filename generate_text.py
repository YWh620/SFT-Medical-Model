from model.model import LoRAFTModel
from transformers import BitsAndBytesConfig
import torch
from data.data_process import load_and_template_local_test_data

base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
lora_dir = "final-model-qwen3-med-lora"


def generate_text():
    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    )

    # Load the LoRA fine-tuned model
    lora_model = LoRAFTModel.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        quantization_config=quantization_config,
        lora_dir=lora_dir
    )

    # Example evaluation input
    test_data = load_and_template_local_test_data('data/test/test.json')
    batch_input_prompts = [
        lora_model.tokenizer.apply_chat_template(sample[:-1], tokenize=False, add_generation_prompt=True) for
        sample in test_data['message']]
    batch_input_ids = lora_model.tokenizer(batch_input_prompts, padding='longest', truncation=True, return_tensors='pt')
    model = lora_model.model
    batch_input_ids = batch_input_ids.to(model.device)
    model.eval()
    with torch.no_grad():
        generated_outputs = model.generate(
            input_ids=batch_input_ids['input_ids'],
            attention_mask=batch_input_ids['attention_mask']
        )
    generated_outputs = generated_outputs[:, batch_input_ids['input_ids'].shape[1]:]
    generated_text = lora_model.re_tokenize(generated_outputs)
    for i, sample in enumerate(test_data['message']):
        print("=== Example {} ===".format(i + 1))
        print("Input Question: {}".format(sample[-2]['content']))
        print("Generated Answer: {}".format(generated_text[i]))
        print()


if __name__ == '__main__':
    print("Starting inference...")
    generate_text()
    print("Training inference.")
