
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def configure_model(model_id, model_cache_dir, use_lora, use_qlora):
    if use_qlora or use_lora:
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            cache_dir= model_cache_dir,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            cache_dir=model_cache_dir,

        )
    return model


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


def get_answer_loss(operation, batch, model, pad_token_id, device="cuda:0"):
    """
    Compute the loss on the answer (i.e., y) part for a multimodal model.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        pad_token_id: The token ID used for padding.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."

    # input_ids, attention_mask, pixel_values, labels = (
    #     batch["input_ids"].to(device),
    #     batch["attention_mask"].to(device),
    #     batch["pixel_values"].to(device),
    #     batch["labels"].to(device),
    # )

    # input_ids, attention_mask, pixel_values, labels = batch
    input_ids, attention_mask, pixel_values, labels = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
        batch[3].to(device),
    )

    # Forward pass through the multimodal model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Shift logits and labels for next token prediction
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    losses = []

    for bid in range(input_ids.shape[0]):
        one_inp = input_ids[bid]

        # Compute the loss per position
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])

        # Negate loss for gradient ascent
        if operation == "ga":
            position_loss = -position_loss

        # Weight loss only for the answer part of the sequence
        position_weight = torch.zeros_like(one_inp)
        position_weight[one_inp == pad_token_id] = 0

        # Normalize weights if applicable
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)

    final_loss = torch.stack(losses).mean()
    return final_loss
