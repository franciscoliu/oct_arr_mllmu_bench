import copy
import json
from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader


class Vanilla_LLaVA_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads JSON data and corresponding images
    from separate directories and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, json_dir: str, image_dir: str, image_base_path: str = None, sort_json_key: bool = True,
                 target_size=None):
        """
        Args:
            json_dir (str): Path to the folder containing JSON files with profile data (text descriptions).
            image_dir (str): Path to the folder containing images.
            image_base_path (str): Base path for the image folder, used in case the 'Directory' field is relative.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
        """
        super().__init__()
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.image_base_path = image_base_path
        self.sort_json_key = sort_json_key
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        # Load the JSON data and image paths and flatten the dataset
        self.dataset = self.flatten_dataset()

    def load_data(self):
        """
        Load the JSON files and pair them with corresponding image paths.
        The dataset is structured as a list of dictionaries, each containing an image path and its associated metadata.
        """
        data = []

        # Loop through all JSON files in the json_dir
        for json_filename in os.listdir(self.json_dir):
            if json_filename.endswith(".json"):
                json_path = os.path.join(self.json_dir, json_filename)

                # Load JSON data
                with open(json_path, 'r') as f:
                    sample = json.load(f)

                # Extract image filename from JSON (assuming it's stored in the 'Directory' field)
                image_filename = sample.get("Directory").split("/")[-1]

                # Get the full image path from image_dir
                image_path = os.path.join(self.image_dir, image_filename)

                # Check if the image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_filename} not found for {json_filename}")
                    continue

                # Append the paired image path and JSON data to the dataset
                data.append({
                    "image_path": image_path,
                    "metadata": sample["metadata"]
                })

        return data

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image path and each QA pair.
        """
        data = self.load_data()
        flattened_data = []

        print("data length", len(data))
        # Flatten each metadata into separate QA pairs
        # for sample in data:
        #     image_path = sample["image_path"]
        #     metadata = sample["metadata"]
        #     for qa_pair in metadata:
        #         question = qa_pair.get("Question", "")
        #         answer = qa_pair.get("Answer", "")
        #         flattened_data.append({
        #             "image_path": image_path,
        #             "question": question,
        #             "answer": answer,
        #         })

        for sample in data:
            image_path = sample["image_path"]
            metadata = sample["metadata"]
            for qa_pair in metadata:
                # Regular question-answer pair
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                if question and answer:
                    flattened_data.append({
                        "image_path": image_path,
                        "question": question,
                        "answer": answer
                    })

                # additional_question = qa_pair.get("Additional_question", "")
                # additional_answer = qa_pair.get("Additional_answer", "")
                # if additional_question and additional_answer:
                #     flattened_data.append({
                #         "image_path": image_path,
                #         "question": additional_question,  # Treat the additional question as a regular question
                #         "answer": additional_answer  # Treat the additional answer as a regular answer
                #     })
        return flattened_data

    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Load the image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            # Resize the image to the target size
            image = self.resize_image(image)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return {}

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer using json2token
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)


        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


class Vanilla_LLaVA_Dataset_baseline(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads JSON data and corresponding images
    from separate directories and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, json_dir: str, image_dir: str, image_base_path: str = None, sort_json_key: bool = True,
                 target_size=None, flatten=False):
        """
        Args:
            json_dir (str): Path to the folder containing JSON files with profile data (text descriptions).
            image_dir (str): Path to the folder containing images.
            image_base_path (str): Base path for the image folder, used in case the 'Directory' field is relative.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            flatten (bool): Whether to flatten the dataset into individual questions. Default is False (work with profiles).
        """
        super().__init__()
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.image_base_path = image_base_path
        self.sort_json_key = sort_json_key
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.flatten = flatten

        # Load the JSON data and either flatten or keep as profiles
        if self.flatten:
            self.dataset = self.flatten_dataset()  # Flattened dataset of individual questions
        else:
            self.dataset = self.load_profiles()  # Dataset of profiles

    def load_profiles(self):
        """
        Load the JSON files, each representing a profile, and do not flatten the data.
        """
        data = []

        for json_filename in os.listdir(self.json_dir):
            if json_filename.endswith(".json"):
                json_path = os.path.join(self.json_dir, json_filename)

                # Load JSON data
                with open(json_path, 'r') as f:
                    sample = json.load(f)

                image_filename = sample.get("Directory").split("/")[-1]
                image_path = os.path.join(self.image_dir, image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_filename} not found for {json_filename}")
                    continue

                # Each profile contains multiple questions (metadata) for a single image
                data.append({
                    "image_path": image_path,
                    "metadata": sample["metadata"]
                })

        return data

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image path and each QA pair.
        """
        data = self.load_profiles()
        flattened_data = []

        for sample in data:
            image_path = sample["image_path"]
            metadata = sample["metadata"]

            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")

                flattened_data.append({
                    "image_path": image_path,
                    "question": question,
                    "answer": answer
                })

        return flattened_data

    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        If flatten=True, returns an individual question-answer pair.
        If flatten=False, returns a profile with multiple questions and the corresponding image.
        """
        sample = self.dataset[idx]

        # Extract the ID from the metadata, assuming it's present in the metadata of the first question
        profile_id = sample["metadata"][0].get("ID", f"Unknown_ID_{idx}")

        # If flatten is False, return the entire profile (with metadata and ID)
        if not self.flatten:
            return {
                "ID": profile_id,
                "image_path": sample["image_path"],
                "metadata": sample["metadata"]
            }

        # If flatten is True, return the individual question-answer pair
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.resize_image(image)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return {}

        question = sample.get("question", "")
        answer = sample.get("answer", "")

        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "ID": profile_id,  # Include the ID field in the output
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


def train_collate_fn(examples, processor, max_length):
    images, texts = [], []
    for image, question, rejected_sequence in examples:
        prompt = f"USER: <image>{question}\nASSISTANT: {rejected_sequence}"
        images.append(image)
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


# def train_collate_fn(examples, processor, max_length):
#     images, texts = [], []
#     for image, question, rejected_sequence in examples:
#         prompt = f"USER: <image>{question}\nASSISTANT: {rejected_sequence}"
#         images.append(image)
#         texts.append(prompt)
#
#     batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     batch["labels"] = labels
#
#     return {
#         "input_ids": batch["input_ids"],
#         "attention_mask": batch["attention_mask"],
#         "pixel_values": batch["pixel_values"],
#         "labels": batch["labels"]
#     }

def eval_collate_fn(examples, processor):
    images, texts, answers = [], [], []
    for image, rejected_sequence in examples:
        prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"
        images.append(image)
        texts.append(prompt)
        answers.append(rejected_sequence)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], answers


# def train_collate_fn_llava(examples, processor, max_length):
#     images = []
#     texts = []
#
#     for example in examples:
#         image = example['image']
#         question = example['question']
#         answer = example['answer']
#
#         images.append(image)
#
#         # Create a prompt where the user asks about the image, and the assistant provides the answer
#         prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
#         texts.append(prompt)
#
#     # Assuming processor is defined, which processes both text and images.
#     batch = processor(
#         text=texts,
#         images=images,
#         padding=True,
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
#
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask out padding tokens in the labels
#
#     batch["labels"] = labels
#
#     input_ids = batch["input_ids"]
#     attention_mask = batch["attention_mask"]
#     pixel_values = batch["pixel_values"]
#     labels = batch["labels"]
#
#     return input_ids, attention_mask, pixel_values, labels

def train_collate_fn_idefics(examples, processor, args):
    """
    A data collator function for LLAVA that processes the input text and images,
    and ensures the number of image tokens matches the number of images.
    """
    texts = []
    images = []

    for example in examples:
        image = example.get("image")
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Create the conversation prompt with the image token placeholder
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]

        # Convert the conversation into a text template
        text = processor.apply_chat_template(messages, add_generation_prompt=False)

        # Append the image and text to respective lists
        texts.append(text.strip())
        images.append([image])


    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")


    # Use the processor to prepare the batch with both text and images
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )

    # Mask labels where the pad token exists in the input_ids
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Assign image token ID to the labels at the appropriate positions
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    labels[labels == processor.tokenizer.pad_token_id] = image_token_id

    batch["labels"] = labels

    if args.trainer:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def train_collate_fn_llava(examples, processor, args):
    # max_length = 384
    # MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    # processor = AutoProcessor.from_pretrained(MODEL_ID)
    # processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        images.append(image)

        # Construct prompt with question and answer
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if args.trainer:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def train_collate_fn_llava_text(examples, tokenizer, args):
    # max_length = 384
    # MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    # processor = AutoProcessor.from_pretrained(MODEL_ID)
    # processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        person_id = example.get('ID')
        print(person_id)
        images.append(image)
        json_file_path = os.path.join("../Full_Set/data_profile/", f"{person_id}.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                profile_data = json.load(json_file)
                # Extract the person's name from the JSON
                name = profile_data.get('biography', {}).get('Name', 'Unknown')
        else:
            name = "Unknown"  # Fallback if the JSON file does not exist

        question_with_name = question.replace("this person", name)
        # Construct prompt with question and answer
        if name == "Unknown":
            prompt = f"USER: This person's name is {name}.\n{question_with_name}\nASSISTANT: {answer}"
        else:
            prompt = f"USER: {question_with_name}\nASSISTANT: {answer}"

        texts.append(prompt)
    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    # Process the batch
    batch = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if args.trainer:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            # "pixel_values": batch["pixel_values"],
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch["labels"]


def eval_collate_fn_llava(examples, processor):
    images = []
    texts = []
    answers = []

    for example in examples:
        image = example['image']
        question = example['question']
        answer = example['answer']

        images.append(image)

        # The model is only provided the prompt, without the answer
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        texts.append(prompt)

        # Keep the ground truth answer separate for evaluation purposes
        answers.append(answer)

    # Process the batch with text and image inputs
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers




chat_template = """
<|image|>
<|begin_of_text|>
### Question:
{user_input}
### Answer:
{assistant_output}
<|end_of_text|>
"""


def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i:i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i:i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs, chat_template=chat_template)
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    print(f"Batch pixel values shape after tokenization: {batch['pixel_values'].shape}")  # Debugging here

    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0

        # Handle prompt headers similar to the example
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1

        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)

        # Mask padding and image token
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:
                labels[i] = -100

        label_list.append(labels)

    batch["labels"] = torch.tensor(label_list)
    return batch

def train_collate_fn_llama(examples, processor, args):
    texts = []
    images = []
    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')

        if image is None:
            raise ValueError("Image not found in the example.")

        # Check if the image is in the correct format (e.g., PIL image or tensor)
        if not isinstance(image, Image.Image):  # Assuming PIL Image format
            raise ValueError("Invalid image format. Expected PIL Image or tensor.")

        # Define the chat template within the collate function
        prompt = f"""<|image|><|begin_of_text|>### Question:{question}### Answer:{answer}<|end_of_text|>"""
        texts.append(prompt)
        images.append(image)  # Append image (ensure it is in the correct format)

    batch = processor(images=images, text=texts, padding=True, return_tensors="pt")

    # Mask the necessary parts of the input and prepare the labels
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.deepcopy(dialog_tokens)

        # Mask logic: handle system/user prompts and assistant headers
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0

        # Handle system and user prompts
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1

        # Handle assistant header
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)

        # Mask padding and image tokens
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:
                labels[i] = -100

        label_list.append(labels)

    # Add the labels to the batch
    batch["labels"] = torch.tensor(label_list)

    return batch


class Vanilla_LLaMA_Dataset(Dataset):
    def __init__(self, json_dir: str, image_dir: str, processor, image_base_path: str = None, sort_json_key: bool = True, target_size=None):
        super().__init__()
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.processor = processor
        self.image_base_path = image_base_path
        self.sort_json_key = sort_json_key
        self.target_size = target_size  # Target size for resizing images
        self.dataset = self.flatten_dataset()

    def load_data(self):
        data = []
        for json_filename in os.listdir(self.json_dir):
            if json_filename.endswith(".json"):
                json_path = os.path.join(self.json_dir, json_filename)
                with open(json_path, 'r') as f:
                    sample = json.load(f)

                image_filename = sample.get("Directory").split("/")[-1]
                image_path = os.path.join(self.image_dir, image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_filename} not found for {json_filename}")
                    continue

                data.append({
                    "image_path": image_path,
                    "metadata": sample["metadata"]
                })
        return data

    def flatten_dataset(self):
        data = self.load_data()
        flattened_data = []

        for sample in data:
            image_path = sample["image_path"]
            metadata = sample["metadata"]
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                if question and answer:
                    flattened_data.append({
                        "image_path": image_path,
                        "question": question,
                        "answer": answer
                    })
        return flattened_data

    def resize_image(self, image):
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Load the image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.resize_image(image)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return {}

        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Create dialog input
        # dialog = [
        #     {"role": "user", "content": [{"type": "text", "text": question}]},
        #     {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        # ]

        # Return the raw dialog and image for the collator to process
        return {"image": image,
                "question": question,
                "answer": answer}
