import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
from utils import get_answer_loss
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset, Dataset
import argparse
import inspect
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration, \
    MllamaForConditionalGeneration, MllamaProcessor, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from data.preprocess import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics, train_collate_fn_llama, Vanilla_LLaMA_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
# from trl import SFTConfig, SFTTrainer
import random
from torch.utils.data import Subset
# previous transformers version: 4.44.2
def convert_to_hf_dataset(custom_dataset):
    data_dict = {
        "image": [],
        "question": [],
        "answer": []
    }

    # Iterate through your custom dataset and extract data
    for idx in range(len(custom_dataset)):
        sample = custom_dataset[idx]
        if sample:  # Check if sample is not empty
            data_dict["image"].append(sample["image"])  # Keep images as paths or tensors, whichever you prefer
            data_dict["question"].append(sample["question"])
            data_dict["answer"].append(sample["answer"])

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data_dict)

    return hf_dataset

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

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def visualize_batch(dataloader, processor, num_samples=3):
    # Get the first batch from the dataloader
    for batch in dataloader:
        input_ids = batch[0][:num_samples]  # Get first num_samples input_ids
        decoded_texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Print the decoded text
        for i, text in enumerate(decoded_texts):
            print(f"\nSample {i + 1} - Decoded Text:")
            print(text)

        break  # Only visualize the first batch


# Example usage:
def load_model_and_processor(model_id):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            cache_dir="CACHE_DIR",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        # Additional processor configuration if necessary
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load LLAVA Next model and processor
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir="CACHE_DIR",
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
        # Additional processor configuration if necessary
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor



######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args.model_id)
    print("Processor Tokenizer Length: ", len(processor.tokenizer)) #128257

    ################## Update #########################
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))


    # Resize token embeddings to match the tokenizer
    # if args.models_id.startswith("meta-llama") == False:
    #     model.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    ################## Update #########################


    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup
    profile_dir = "Train_data/full_train"
    image_base_path = "Full_Set/data_original"
    dataset = Vanilla_LLaVA_Dataset(json_dir=profile_dir, image_dir=image_base_path)
    # Training loop
    if args.trainer:
        train_dataset = convert_to_hf_dataset(dataset)
        if args.model_id.startswith("llava"):
            data_collator = lambda x: train_collate_fn_llava(x, processor, args)
        elif args.model_id.startswith("HuggingFaceM4"):
            data_collator = lambda x: train_collate_fn_idefics(x, processor, args)
        else:
            raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
        training_args = TrainingArguments(
                output_dir=args.save_dir,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=4 if args.gradient_accumulation else 1,
                learning_rate=args.lr,
                warmup_steps=0,
                logging_steps=10,
                evaluation_strategy="no",  # You can change this if you have a validation set
                save_steps=500,  # Save model every 500 steps
                save_strategy="no",
                save_total_limit=1,
            # save_total_limit=3,  # Limit total number of model checkpoints
                remove_unused_columns=False,
                # fp16=True,  # Enable mixed precision for faster training if needed
            )

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataset) * args.num_epochs // args.batch_size,
        )

        # Use HuggingFace Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
        )

        # Start training
        trainer.train()

        # trainer.save_model(args.save_dir)
        model = model.merge_and_unload()
        # # Save the final model
        model.save_pretrained(args.save_dir)
        print(f"Model saved to: {args.save_dir}")
    else:
        if args.model_id.startswith("llava"):
            train_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            train_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
            )
        else:
            raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

        # Accelerator setup
        accelerator = Accelerator()
        if args.gradient_accumulation:
            print("Gradient accumulation enabled.")
            accumulation_steps = 4  # Adjust based on memory
            model.gradient_checkpointing_enable()
        else:
            print("Gradient accumulation disabled.")

        optimizer = AdamW(model.parameters(), lr=args.lr)

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * args.num_epochs,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )


        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")


            if args.gradient_accumulation:
                for step, batch in enumerate(progress_bar):
                    if args.model_id.startswith("meta-llama"):
                        # print(f"Batch keys: {batch.keys()}")
                        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                        # print(f"Batch: {batch}")
                        with accelerator.accumulate(model):
                            outputs = model(**batch)
                            loss = outputs.loss / accumulation_steps
                            # loss.requires_grad = True
                            # Debugging requires_grad for loss
                            print(f"Loss requires grad: {loss.requires_grad}")  # Should be True
                            print(f"Loss grad_fn: {loss.grad_fn}")  # Should not be None
                            print("Loss: ", loss)
                            accelerator.backward(loss)
                            if (step + 1) % accumulation_steps == 0:
                                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                                optimizer.step()
                                optimizer.zero_grad()
                                lr_scheduler.step()
                    else:
                        input_ids, attention_mask, pixel_values, labels = batch
                        with accelerator.accumulate(model):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                            pixel_values=pixel_values, labels=labels)
                            loss = outputs.loss / accumulation_steps
                            # loss = outputs.loss
                            accelerator.backward(loss)
                            if (step + 1) % accumulation_steps == 0:
                                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                                optimizer.step()
                                optimizer.zero_grad()
                                lr_scheduler.step()
                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=total_loss / len(progress_bar))
                # lr_scheduler.step()
                print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

            else:
                for batch in progress_bar:
                    if args.model_id.startswith("meta-llama"):
                        # Unpack the batch to get input_ids, attention_mask, pixel_values, aspect_ratio_ids, and labels
                        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                        outputs = model(**batch)
                        # Compute the loss
                        loss = outputs.loss

                        # Debugging info: Ensure the model output is valid
                        # print(f"Loss requires grad: {loss.requires_grad}")  # Should be True
                        # print(f"Loss grad_fn: {loss.grad_fn}")  # Should not be None
                        # print(f"Loss value: {loss.item()}")

                        # Backward pass and optimization
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                        # Track the total loss
                        total_loss += loss.item()
                        progress_bar.set_postfix(loss=total_loss / len(progress_bar))
                    else:
                        input_ids, attention_mask, pixel_values, labels = batch
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        pixel_values=pixel_values,
                                        labels=labels)
                        loss = outputs.loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        total_loss += loss.item()
                        progress_bar.set_postfix(loss=total_loss / len(progress_bar))
                print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

        # Save the final model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # if args.model_id.startswith("meta-llama") == False:
        unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(args.save_dir)
        print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

    # Call main function
    main(args)

