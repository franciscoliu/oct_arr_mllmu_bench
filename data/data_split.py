import os
import sys
from collections import defaultdict, Counter

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset, Dataset
import random
import torch
import os
import json
from torch.utils.data import Subset
import argparse
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from data.preprocess import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics, Vanilla_LLaVA_Dataset_baseline
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


def append_or_replace_qa(source_folder, destination_folder):
    """
    Append or replace the last question-answer pair from each source JSON file to the corresponding destination JSON file.

    Args:
        source_folder (str): Path to the folder containing the source JSON files (e.g., "001.json", "002.json").
        destination_folder (str): Path to the folder containing the destination JSON files (e.g., "001_qa.json", "002_qa.json").
    """

    # Loop through each source file in the source folder
    for source_filename in sorted(os.listdir(source_folder)):
        if source_filename.endswith(".json"):
            # Get the person ID (e.g., "001" from "001.json")
            person_id = source_filename.split(".")[0]

            # Construct the corresponding destination filename (e.g., "001_qa.json")
            destination_filename = f"{person_id}_qa.json"

            # Construct the full file paths for the source and destination files
            source_file_path = os.path.join(source_folder, source_filename)
            destination_file_path = os.path.join(destination_folder, destination_filename)

            # Check if the destination file exists
            if not os.path.exists(destination_file_path):
                print(f"Destination file {destination_file_path} not found. Skipping.")
                continue

            # Load the source JSON file
            with open(source_file_path, 'r') as source_file:
                source_data = json.load(source_file)

            # Load the destination JSON file
            with open(destination_file_path, 'r') as destination_file:
                destination_data = json.load(destination_file)

            # Extract the last question-answer pair from the source JSON
            last_question = source_data.get("question")
            last_answer = source_data.get("answer")
            additional_qa = {
                "Additional_ID": person_id,
                "Additional_question": last_question,
                "Additional_answer": last_answer
            }

            # Check if the entry with "Additional_ID" already exists in the destination's "metadata" list
            existing_index = next((index for (index, d) in enumerate(destination_data["metadata"])
                                   if d.get("Additional_ID") == person_id), None)

            if existing_index is not None:
                # If the entry exists, replace it
                destination_data["metadata"][existing_index] = additional_qa
                print(f"Replaced QA pair for ID {person_id} in {destination_filename}")
            else:
                # If the entry does not exist, append the new QA pair
                destination_data["metadata"].append(additional_qa)
                print(f"Appended QA pair for ID {person_id} in {destination_filename}")

            # Save the updated destination JSON file, ensuring non-ASCII characters are not escaped
            with open(destination_file_path, 'w') as destination_file:
                json.dump(destination_data, destination_file, ensure_ascii=False, indent=4)


def update_json_id(json_folder):
    # Loop through all files in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            # Load the JSON file
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Get the global "ID"
            global_id = data.get("ID")

            # Update the "ID" field inside each metadata entry
            for entry in data.get("metadata", []):
                entry["ID"] = global_id  # Set the entry "ID" to match the global "ID"

            # Save the updated JSON back to the file
            with open(json_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Updated {filename}")
def find_profiles_with_details(json_dir, output_file):
    """
    Finds profiles with duplicate names and records their IDs, and counts the number of occurrences of jobs and born places.

    Args:
        json_dir (str): Directory containing the profile JSON files.
        output_file (str): Path to the output JSON file to save the results.
    """
    # Dictionary to hold names, employment types, birthplaces with associated IDs
    name_to_ids = defaultdict(list)
    employment_counter = Counter()
    born_place_counter = Counter()

    # Iterate over all JSON files in the directory
    for json_filename in os.listdir(json_dir):
        if json_filename.endswith(".json"):
            json_path = os.path.join(json_dir, json_filename)

            # Load the JSON data
            with open(json_path, 'r') as f:
                profile_data = json.load(f)

            # Extract relevant fields
            profile_id = profile_data.get("ID", "Unknown_ID")
            biography = profile_data.get("biography", {})

            name = biography.get("Name", "Unknown_Name")
            employment = biography.get("Employment", "Unknown_Employment")
            born_place = biography.get("Born", "Unknown_Born_Place")

            # Add the ID to the name's list (for duplicates check)
            name_to_ids[name].append(profile_id)

            # Count the number of occurrences of each job and born place
            employment_counter[employment] += 1
            born_place_counter[born_place] += 1

    # Only keep names that are duplicated (i.e., appear more than once)
    duplicated_names = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}

    # Prepare the final result to be saved
    results = {
        "duplicated_names": duplicated_names,  # Only save duplicated names and their IDs
        "employment_summary": employment_counter,  # Summary count of employment types
        "born_place_summary": born_place_counter  # Summary count of birth places
    }

    # Save the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")



def flatten_dataset(profiles):
    """
    Flatten the profiles dataset such that each question-answer pair becomes a single sample.
    Args:
        profiles (Subset or list): List of profiles or a subset of profiles.
    Returns:
        flattened_data (list): List of dictionaries with image path and each QA pair.
    """
    flattened_data = []
    for profile in profiles:
        # Each profile has an image path and metadata (question-answer pairs)
        image_path = profile["image_path"]
        # print(image_path)
        # print(profile)
        # Ensure the image is loaded correctly (do this in the flattening step)
        try:
            image = Image.open(image_path).convert("RGB")  # Load the image as PIL
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            continue  # Skip profiles with invalid images

        # Flatten the metadata (which contains the questions and answers)
        metadata = profile["metadata"]
        for qa_pair in metadata:
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")

            flattened_data.append({
                "image": image,  # Include the image here
                "question": question,
                "answer": answer
            })

    return flattened_data


def split_and_save_dataset(dataset, save_dir, split_ratios):
    """
    Split the dataset based on profiles (not questions) and save the forget/retain sets, along with the selected IDs.
    Args:
        dataset: The Vanilla_LLaVA_Dataset_baseline, which contains profiles.
        save_dir: Directory to save the forget and retain datasets.
        split_ratios: List of percentages for splitting (e.g., [5, 10, 15]).
    """
    dataset_size = len(dataset)  # Number of profiles
    all_indices = list(range(dataset_size))  # Get indices for all profiles

    for forget_percentage in split_ratios:
        retain_percentage = 100 - forget_percentage

        forget_dir = os.path.join(save_dir, f"forget_{forget_percentage}")
        retain_dir = os.path.join(save_dir, f"retain_{retain_percentage}")

        forget_path = os.path.join(forget_dir, "forget_dataset.pt")
        retain_path = os.path.join(retain_dir, "retain_dataset.pt")
        forget_ids_path = os.path.join(forget_dir, "forget_ids.json")
        retain_ids_path = os.path.join(retain_dir, "retain_ids.json")

        if os.path.exists(forget_path) and os.path.exists(retain_path):
            print(f"Split for forget {forget_percentage}% and retain {retain_percentage}% already exists, skipping.")
            continue

        print(f"Creating split for forget {forget_percentage}% and retain {retain_percentage}%.")

        forget_size = int(forget_percentage / 100 * dataset_size)
        retain_size = dataset_size - forget_size

        # Randomize the indices to ensure a random split
        random.shuffle(all_indices)
        forget_indices = all_indices[:forget_size]
        retain_indices = all_indices[forget_size:forget_size + retain_size]

        forget_dataset = Subset(dataset, forget_indices)
        retain_dataset = Subset(dataset, retain_indices)

        # Debugging: Print metadata for the forget set
        print("Metadata for forget set:")
        for idx in forget_indices:
            print(dataset[idx]["metadata"])  # Print metadata of each selected profile for debugging
        print("\n")

        # Save the IDs of the selected forget and retain samples
        forget_ids = [dataset[idx]["ID"] for idx in forget_indices]  # Extract the ID field from each sample
        retain_ids = [dataset[idx]["ID"] for idx in retain_indices]  # Extract the ID field from each sample

        # Create the directories
        os.makedirs(forget_dir, exist_ok=True)
        os.makedirs(retain_dir, exist_ok=True)

        # Save datasets
        torch.save(forget_dataset, forget_path)
        torch.save(retain_dataset, retain_path)

        # Save the IDs in JSON format
        with open(forget_ids_path, 'w') as f:
            json.dump(forget_ids, f, indent=4)
        with open(retain_ids_path, 'w') as f:
            json.dump(retain_ids, f, indent=4)

        print(f"Saved forget {forget_percentage}% dataset and IDs to {forget_dir}")
        print(f"Saved retain {retain_percentage}% dataset and IDs to {retain_dir}")


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

def load_dataset_split(data_split_dir, split_ratio):
    """
    Load the corresponding forget and retain datasets based on the split_ratio argument.

    Args:
        data_split_dir (str): Base directory where the split datasets are stored.
        split_ratio (int): Percentage of forget set (e.g., 5, 10, 15).

    Returns:
        forget_dataset (torch.utils.data.Dataset): The forget dataset.
        retain_dataset (torch.utils.data.Dataset): The retain dataset.
    """
    # Define the corresponding folder names
    forget_dir = os.path.join(data_split_dir, f"forget_{split_ratio}")
    retain_dir = os.path.join(data_split_dir, f"retain_{100 - split_ratio}")

    # Define the paths to the saved datasets
    forget_path = os.path.join(forget_dir, "forget_dataset.pt")
    retain_path = os.path.join(retain_dir, "retain_dataset.pt")

    # Check if the paths exist and load the datasets
    if os.path.exists(forget_path) and os.path.exists(retain_path):
        print(f"Loading forget {split_ratio}% dataset from {forget_path}")
        forget_dataset = torch.load(forget_path)

        print(f"Loading retain {100 - split_ratio}% dataset from {retain_path}")
        retain_dataset = torch.load(retain_path)
    else:
        raise FileNotFoundError(
            f"Could not find forget or retain dataset for split ratio {split_ratio} in {data_split_dir}")

    return forget_dataset, retain_dataset

def main(args):
    source_folder = "../Full_Set/data_profile"  # Folder containing "001.json", "002.json", etc.
    destination_folder = "../Train_data/full_train"  # Folder containing "001_qa.json", "002_qa.json", etc.

    append_or_replace_qa(source_folder, destination_folder)

    profile_dir = "../Train_data/full_train"
    image_base_path = "../Full_Set/data_original"
    dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    print(f"Dataset size (profiles): {len(dataset)}")

    ############################################### Debug Purpose ###############################################
    # sample_size = 10  # Adjust this number as needed
    # subset_indices = list(range(sample_size))
    #
    # # Use a Subset to create a smaller dataset
    # subset_dataset = Subset(dataset, subset_indices)
    # print(f"Subset size: {len(subset_dataset)}")
    #
    # for idx in subset_indices:
    #     sample = dataset[idx]  # Access the sample directly from the full dataset
    #     print(sample)
    ############################################### Debug Purpose ###############################################

    # Split the dataset based on profiles
    split_ratios = [5, 10, 15]
    split_and_save_dataset(dataset, args.data_split_dir, split_ratios)

    # Load the forget and retain sets (based on profiles)
    forget_dataset, retain_dataset = load_dataset_split(args.data_split_dir, args.forget_split_ratio)

    print(f"Forget dataset size (profiles): {len(forget_dataset)}")
    print(f"Retain dataset size (profiles): {len(retain_dataset)}")
    # Flatten the datasets into individual question-answer pairs
    flattened_forget_dataset = flatten_dataset(forget_dataset)
    flattened_retain_dataset = flatten_dataset(retain_dataset)

    # Print flattened dataset sizes
    print(f"Flattened forget dataset size (questions): {len(flattened_forget_dataset)}")
    print(f"Flattened retain dataset size (questions): {len(flattened_retain_dataset)}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")

    args = parser.parse_args()

    # Call main function
    main(args)
