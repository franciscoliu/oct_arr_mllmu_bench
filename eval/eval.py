
import os
import json
import random

from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration, MllamaProcessor, MllamaForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score

def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
    """
    Load IDs from the JSON file and find their corresponding evaluation question files with a specific filename pattern,
    then return a list of the loaded JSON files.

    Args:
        id_json_file (str): Path to the JSON file containing the list of IDs.
        question_folder (str): Path to the folder containing evaluation question files.
        filename_pattern (str): Filename pattern to match (e.g., "*_question.json"). Default is "*" for any file.

    Returns:
        list: A list of loaded JSON files from the question folder.
    """
    # Load the list of IDs from the ID JSON file
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []

    # Loop through the files in the question folder
    for filename in sorted(os.listdir(question_folder)):
        # Find files that match the ID and the filename pattern
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)

                # Load the matching JSON file
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break  # Move to the next file after finding the match

    return json_files

def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt


def formulate_prompt_with_options_llama(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n####Choices:\n{options_str}"
    return prompt
def split_dataset(original_dataset, forget_percentage=0.3):
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
    """
    Load all JSON files from the given folder.
    """
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
    """
    Load an image, trying both .png and .jpg extensions.
    """
    possible_extensions = ['.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None

def evaluate_classification(json_files, image_folder, processor, tokenizer, model, args, id_list_file=None, mode="default"):
    """
    Evaluate classification task with/without few-shot samples based on the mode.

    Args:
        json_files: List of JSON data files containing questions and answers.
        image_folder: Folder containing images.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Mode that controls how few-shot samples are handled ('forget', 'retain_share', or others). Default is 'default'.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print(
        "################################## Classification Task Starts ##############################################")

    few_shot_dir = "../Full_Set/classification_qa"  # Directory containing few-shot samples

    # Load the id_list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    else:
        # If no id_list_file is provided, extract IDs from the json_files
        id_list = [json_file["ID"] for json_file in json_files]

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'json_files'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Randomly select 3 IDs from the id_list for few-shot samples

    if args.model_id.startswith("HuggingFaceM4"):
        selected_ids = random.sample(id_list, 1)
    elif args.model_id.startswith("llava"):
        selected_ids = random.sample(id_list, 2)
    elif args.model_id.startswith("meta-llama"):
        selected_ids = random.sample(id_list, 1)

    print(f"Selected few-shot IDs: {selected_ids}")

    few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
    few_shot_images = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}  # Store which questions are used for few-shot learning for each ID

    if mode in ["forget", "retain_shared", "test"]:
        print(f"Few-shot mode: {mode}")

        # Load the few-shot samples based on selected_ids
        for selected_id in selected_ids:
            # Find the json file for the selected ID in the few_shot_dir
            json_path = os.path.join(few_shot_dir, f"{selected_id}_questions.json")

            if not os.path.exists(json_path):
                print(f"Skipping {json_path}, file not found.")
                continue

            with open(json_path, 'r') as f:
                json_file = json.load(f)

            image_textual_questions = json_file.get("Image_Textual_Questions", [])
            pure_text_questions = json_file.get("Pure_Text_Questions", [])

            # Track which questions are used for few-shot learning
            few_shot_question_indices[selected_id] = {
                "image_textual": 0 if image_textual_questions else None,
                "pure_text": 0 if pure_text_questions else None
            }

            if image_textual_questions:
                first_image_textual_question = image_textual_questions[0]
                few_shot_image_prompts.append({
                    "Question": first_image_textual_question["Question"],
                    "Options": first_image_textual_question["Options"],
                    "Answer": first_image_textual_question["Correct_Answer"]
                })

                # Load the corresponding image
                image_id = json_file.get("ID")
                image = load_image(image_folder, image_id)

                if image is not None:
                    few_shot_images.append(image)
                else:
                    print(f"Skipping image for ID: {image_id} (image not found)")

            if pure_text_questions:
                first_pure_text_question = pure_text_questions[0]
                few_shot_text_prompts.append({
                    "Question": first_pure_text_question["Question"],
                    "Options": first_pure_text_question["Options"],
                    "Answer": first_pure_text_question["Correct_Answer"]
                })

    else:
        print(f"Normal mode: {mode}, no few-shot examples applied.")
        # Skip setting up few-shot examples and just proceed with normal evaluation.

    # Now, process the actual classification samples in json_files
    for idx, json_file in enumerate(tqdm(json_files), start=1):
        if mode == "retain_shared" and json_file["ID"] in selected_ids:
            # Skip this entire sample because it was used for few-shot
            print(f"Skipping sample {json_file['ID']} as it was used for few-shot learning.")
            continue

        image_textual_questions = json_file.get("Image_Textual_Questions", [])
        pure_text_questions = json_file.get("Pure_Text_Questions", [])
        image_id = json_file.get("ID")

        # Load the image for the current sample
        image = load_image(image_folder, image_id)
        if image is None:
            print(f"Skipping image-based questions for image: {image_id}")
            continue

        # Process Image_Textual_Questions
        print("########################## Processing Image-Textual Questions ########################## ")
        for question_idx, question_data in enumerate(image_textual_questions, start=1):
            # Skip the few-shot questions if in 'retain_shared' mode
            if mode == "retain_shared" and json_file["ID"] in selected_ids and \
                    few_shot_question_indices[json_file["ID"]]["image_textual"] == question_idx - 1:
                print(f"Skipping few-shot image-textual question {question_idx} for ID: {json_file['ID']}")
                continue

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            # Prepare the few-shot prompt if applicable (only when mode is 'forget' or 'retain_share')
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for i, few_shot_image in enumerate(few_shot_images):
                    few_shot_question = few_shot_image_prompts[i]["Question"]
                    few_shot_options = few_shot_image_prompts[i]["Options"]
                    few_shot_answer = few_shot_image_prompts[i]["Answer"]
                    if args.model_id.startswith("meta-llama"):
                        few_shot_prompt += (
                            f"<|image|>\n"  # Insert image placeholder for each few-shot image
                            f"<|begin_of_text|>### Question: {few_shot_question}\n"
                            f"Options:\n"
                            f"A: {few_shot_options['A']}\n"
                            f"B: {few_shot_options['B']}\n"
                            f"C: {few_shot_options['C']}\n"
                            f"D: {few_shot_options['D']}\n"
                            f"Correct Answer: {few_shot_answer}\n"
                            # f"<|end_of_text|>\n"
                        )
                    else:
                        few_shot_prompt += (
                            f"USER: <image>\n"
                            f"Question: {few_shot_question}\n"
                            f"A: {few_shot_options['A']}\n"
                            f"B: {few_shot_options['B']}\n"
                            f"C: {few_shot_options['C']}\n"
                            f"D: {few_shot_options['D']}\n"
                            f"Correct Answer: {few_shot_answer}\n"
                        )

            # Combine few-shot examples with the current prompt (only if mode requires it)
            # Preprocess the image and prompt
            if args.model_id.startswith("HuggingFaceM4"):
                prompt = (f"{few_shot_prompt}"
                          f"USER: <image>\n"
                          f"{question_with_options}\n"
                          f"Just give ONE letter representing the answer directly.\nASSISTANT:")
                if mode in ["forget", "retain_shared","test"]:
                    inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                else:
                    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

                # generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            elif args.model_id.startswith("llava"):
                prompt = (f"{few_shot_prompt}"
                          f"USER: <image>\n"
                          f"{question_with_options}\n"
                          f"Answer with the option's letter from the given choices directly.\nASSISTANT:")

                if mode in ["forget", "retain_shared","test"]:
                    inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                else:
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            elif args.model_id.startswith("meta-llama"):
                question_with_options = formulate_prompt_with_options_llama(question, options)
                few_shot_prompt =""
                prompt = (f"{few_shot_prompt}"
                          f"<|image|><|begin_of_text|> ###Instruction: You are a helpful assistant that will answer all following multiple choice questions in letter from A, B, C, D \n"
                          f"###Question: {question_with_options}\n"
                          f"###Answer:")

                if mode in ["forget", "retain_shared", "test"]:
                    # inputs = processor(images=[*few_shot_images, image], text=prompt, padding=True,
                    #                    return_tensors="pt").to("cuda")
                    inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to("cuda")
                else:
                    inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

                generated_text = processor.decode(outputs[0], skip_special_tokens=True)


            print("Generated Text: ", generated_text)
            print("\n")

            # # Process the answer
            # assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            # predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None

            # Process the answer
            if "ASSISTANT:" in generated_text:
                # If "ASSISTANT:" is present, split and process accordingly
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_text:
                # If "Answer:" is present, split and process accordingly
                assistant_response = generated_text.split("Answer:")[1].strip()
            else:
                # Fallback to the entire generated text if neither "ASSISTANT:" nor "Answer:" is found
                assistant_response = generated_text.strip()

            # Extract the first character as the predicted answer, and ensure it's a valid option
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None


            print("Prompt: ", prompt)
            print(f"Model Answer for image-text: {predicted_answer}")
            print(f"Correct Answer: {correct_answer}\n")

            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1

        # Process Pure_Text_Questions
        print("######################### Processing Pure Text Questions #########################")
        for question_idx, question_data in enumerate(pure_text_questions, start=1):
            # Skip the few-shot questions if in 'retain_share' mode
            if mode == "retain_shared" and json_file["ID"] in selected_ids and \
                    few_shot_question_indices[json_file["ID"]]["pure_text"] == question_idx - 1:
                print(f"Skipping few-shot pure-text question {question_idx} for ID: {json_file['ID']}")
                continue

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            # Construct the few-shot prompt
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for few_shot in few_shot_text_prompts:
                    few_shot_question = few_shot["Question"]
                    few_shot_options = few_shot["Options"]
                    few_shot_answer = few_shot["Answer"]
                    if args.model_id.startswith("meta-llama"):
                        few_shot_prompt += (
                            f"<|begin_of_text|>### Question: {few_shot_question}\n"
                            f"Options:\n"
                            f"A: {few_shot_options['A']}\n"
                            f"B: {few_shot_options['B']}\n"
                            f"C: {few_shot_options['C']}\n"
                            f"D: {few_shot_options['D']}\n"
                            f"Correct Answer: {few_shot_answer}\n"
                            # f"<|end_of_text|>\n"
                        )
                    else:
                        few_shot_prompt += (
                            f"USER:\n"
                            f"Question: {few_shot_question}\n"
                            f"A: {few_shot_options['A']}\n"
                            f"B: {few_shot_options['B']}\n"
                            f"C: {few_shot_options['C']}\n"
                            f"D: {few_shot_options['D']}\n"
                            f"Correct Answer: {few_shot_answer}\n"
                        )

            # Formulate the full prompt with the few-shot examples and the current question

            # Decode the model output to readable text
            if args.model_id.startswith("HuggingFaceM4"):
                prompt = (
                    f"{few_shot_prompt}"  # Few-shot prompt
                    f"USER:\n{question_with_options}\n"
                    f"Just give ONE letter representing the answer directly.\nASSISTANT:"
                )
                # Preprocess the input
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
                # generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            elif args.model_id.startswith("llava"):
                prompt = (
                    f"{few_shot_prompt}"  # Few-shot prompt
                    f"USER:\n{question_with_options}\n"
                    f"Answer with the option's letter from the given choices directly.\nASSISTANT:"
                )
                # Preprocess the input
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            elif args.model_id.startswith("meta-llama"):
                question_with_options = formulate_prompt_with_options_llama(question, options)
                few_shot_prompt =""
                prompt = (
                    f"{few_shot_prompt}"  # Few-shot prompt
                    f"<|begin_of_text|>### Question: {question_with_options}\n"
                    f"Please only generate ONE letter representing the answer!\n### Answer:"
                )
                # Preprocess the input
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Map model output to one of the options
            print("Generated Text: ", generated_text)
            print("\n")

            # Map model output to one of the options
            # assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            # predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None

            # Process the answer
            if "ASSISTANT:" in generated_text:
                # If "ASSISTANT:" is present, split and process accordingly
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_text:
                # If "Answer:" is present, split and process accordingly
                assistant_response = generated_text.split("Answer:")[1].strip()
            else:
                # Fallback to the entire generated text if neither "ASSISTANT:" nor "Answer:" is found
                assistant_response = generated_text.strip()

            # Extract the first character as the predicted answer, and ensure it's a valid option
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None

            print("Prompt:", prompt)
            print(f"Model Answer for pure text: {predicted_answer}")
            print(f"Correct Answer: {correct_answer}\n")

            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy
    }


def evaluate_generation(json_files, image_folder, processor, tokenizer, model, args, file_name):
    """
    Evaluate the generation tasks using the ROUGE and BLEU scores.

    Args:
        json_files (list): List of JSON data files containing the questions and ground truth.
        image_folder (str): Path to the folder containing images.
        processor: The processor for handling text and images (e.g., from Hugging Face).
        model: The model for answering the generation questions.
        output_file (str): Path to save the evaluation results.

    Returns:
        dict: A dictionary containing average ROUGE and BLEU scores for Image_Textual and Pure_Text questions.
    """
    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Loop through each person's data
    for person_data in tqdm(json_files):
        image_id = person_data.get("ID")
        image = load_image(image_folder, image_id)

        # Process each generation question
        for question_data in person_data["Generation_Questions"]:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH!!.\nASSISTANT: "

                if args.model_id.startswith("HuggingFaceM4"):
                    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("llava"):
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=llama_prompt, return_tensors="pt")
                else:
                    ValueError("Model ID not supported")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            else:  # Pure_Text case
                if args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|begin_of_text|>### Question: {question}\n### Answer:"
                    inputs = processor(text=llama_prompt, return_tensors="pt").to("cuda")
                else:
                    prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH!!\nASSISTANT:"
                    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
            if "ASSISTANT:" in generated_answer:
                predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_answer:
                predicted_answer = generated_answer.split("Answer:")[1].strip()
            else:
                predicted_answer = generated_answer.strip()

            # Print debug information
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                # Accumulate scores for Image_Textual questions
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                # Accumulate scores for Pure_Text questions
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{file_name}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print separate ROUGE scores for Image_Textual questions
    if total_image_textual_questions > 0:
        avg_rouge1_img = total_rouge1_img / total_image_textual_questions
        avg_rouge2_img = total_rouge2_img / total_image_textual_questions
        avg_rougeL_img = total_rougeL_img / total_image_textual_questions
        avg_bleu_img = total_bleu_img / total_image_textual_questions
        print(f"Average ROUGE-1 (Image_Textual): {avg_rouge1_img}")
        print(f"Average ROUGE-2 (Image_Textual): {avg_rouge2_img}")
        print(f"Average ROUGE-L (Image_Textual): {avg_rougeL_img}")
        print(f"Average BLEU (Image_Textual): {avg_bleu_img}")

    # Print separate ROUGE scores for Pure_Text questions
    if total_pure_text_questions > 0:
        avg_rouge1_text = total_rouge1_text / total_pure_text_questions
        avg_rouge2_text = total_rouge2_text / total_pure_text_questions
        avg_rougeL_text = total_rougeL_text / total_pure_text_questions
        avg_bleu_text = total_bleu_text / total_pure_text_questions
        print(f"Average ROUGE-1 (Pure_Text): {avg_rouge1_text}")
        print(f"Average ROUGE-2 (Pure_Text): {avg_rouge2_text}")
        print(f"Average ROUGE-L (Pure_Text): {avg_rougeL_text}")
        print(f"Average BLEU (Pure_Text): {avg_bleu_text}")

    # Return the results as a dictionary
    return {
        "Average ROUGE-1 (Image_Textual)": avg_rouge1_img if total_image_textual_questions > 0 else 0,
        "Average ROUGE-2 (Image_Textual)": avg_rouge2_img if total_image_textual_questions > 0 else 0,
        "Average ROUGE-L (Image_Textual)": avg_rougeL_img if total_image_textual_questions > 0 else 0,
        "Average BLEU (Image_Textual)": avg_bleu_img if total_image_textual_questions > 0 else 0,
        "Average ROUGE-1 (Pure_Text)": avg_rouge1_text if total_pure_text_questions > 0 else 0,
        "Average ROUGE-2 (Pure_Text)": avg_rouge2_text if total_pure_text_questions > 0 else 0,
        "Average ROUGE-L (Pure_Text)": avg_rougeL_text if total_pure_text_questions > 0 else 0,
        "Average BLEU (Pure_Text)": avg_bleu_text if total_pure_text_questions > 0 else 0,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')

    parser.add_argument('--forget_image_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--test_image_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--celebrity_image_folder', type=str, required=True, help='Path to real person image folder.')


    parser.add_argument('--celebrity_classification_question_folder', type=str, required=True,
                        help='Path to real person classification question folder.')
    parser.add_argument('--celebrity_generation_question_folder', type=str, required=True,
                        help='Path to real person generation question folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument('--data_split_folder', type=str, default="../Data_split", help='Path to real person image folder.')
    parser.add_argument('--pretrain', type=bool, default=False, help="load pretrain model")


    return parser.parse_args()

def main():
    args = parse_arguments()
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Load IDs from the forget folder
    forget_id_json_file = os.path.join(forget_folder,'forget_ids.json')  # assuming 'forget_ids.json' exists in the folder
    forget_classification_set = evaluate_from_ids(forget_id_json_file, "../Full_Set/classification_qa", "*_questions.json")
    forget_generation_set = evaluate_from_ids(forget_id_json_file, "../Full_Set/generation_qa", "*_generation_questions.json")

    test_classification_set = evaluate_from_ids(forget_id_json_file, "../Test_Set/classification_qa","*_questions.json")
    test_generation_set = evaluate_from_ids(forget_id_json_file, "../Test_Set/generation_qa","*_generation_questions.json")

    # Load IDs from the retain folder
    retain_id_json_file = os.path.join(retain_folder,'retain_ids.json')  # assuming 'retain_ids.json' exists in the folder
    retain_classification_set = evaluate_from_ids(retain_id_json_file, "../Full_Set/classification_qa","*_questions.json")
    retain_generation_set = evaluate_from_ids(retain_id_json_file, "../Full_Set/generation_qa","*_generation_questions.json")

    # Load real person data for retain set
    real_person_classification_questions = load_json_files(args.celebrity_classification_question_folder)
    real_person_generation_questions = load_json_files(args.celebrity_generation_question_folder)

    print("Length of forget_classification_set",len(forget_classification_set))
    print("Length of forget_generation_set",len(forget_generation_set))
    print("Length of test_classification_set", len(test_classification_set))
    print("Length of test_generation_set", len(test_generation_set))
    print("Length of retain_classification_set",len(retain_classification_set))
    print("Length of retain_generation_set",len(retain_generation_set))
    print("Length of real_person_classification_questions",len(real_person_classification_questions))
    print("Length of real_person_generation_questions",len(real_person_generation_questions))

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()
    if args.pretrain:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Pretrained model...")
            # Load LLAVA model and processor
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                # torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                # quantization_config=bnb_config,
                cache_dir="CACHE_DIR",
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print("Loading idefics2 Pretrained model...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                "HuggingFaceM4/idefics2-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                cache_dir="CACHE_DIR",
            )

    else:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Vanilla model...")
            model = LlavaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print("Loading idefics2 Vanilla model...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # load_in_4bit=True,
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif args.model_id.startswith("meta-llama"):
            print("Loading Meta-LLAMA Vanilla model...")
            model = MllamaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                local_files_only=True
            )

    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    # print("### Evaluating Forget Set ###")
    forget_classification_results = evaluate_classification(forget_classification_set, args.forget_image_folder, processor,tokenizer, model, args, retain_id_json_file,"forget")
    forget_generation_results = evaluate_generation(forget_generation_set, args.forget_image_folder, processor, tokenizer, model, args, "forget")

    # torch.cuda.empty_cache()
    print("### Evaluating Test Set ###")
    test_classification_results = evaluate_classification(test_classification_set, args.test_image_folder, processor, tokenizer, model, args, retain_id_json_file,"test")
    test_generation_results = evaluate_generation(test_generation_set, args.test_image_folder, processor,tokenizer, model, args, "test")

    # Evaluate Retain Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    print("### Evaluating Retain Set (from shared original dataset) ###")
    retain_classification_results = evaluate_classification(retain_classification_set, args.forget_image_folder, processor,tokenizer, model, args, retain_id_json_file, "retain_shared")
    retain_generation_results = evaluate_generation(retain_generation_set, args.forget_image_folder, processor, tokenizer, model, args, "retain_shared")

    # Evaluate Retain Set (real person knowledge)
    torch.cuda.empty_cache()
    print("### Evaluating Retain Set (real person knowledge) ###")
    # real_person_classification_results = ""
    # real_person_generation_results = ""
    real_person_classification_results = evaluate_classification(real_person_classification_questions,
                                                                 args.celebrity_image_folder, processor,tokenizer, model, args, retain_id_json_file,"retain_celebrity")
    real_person_generation_results = evaluate_generation(real_person_generation_questions,
                                                         args.celebrity_image_folder, processor,tokenizer, model, args, "retain_celebrity")

    # Output results
    print("Forget Set Results:")
    print(forget_classification_results)
    print(forget_generation_results)

    print("Test Set Results:")
    print(test_classification_results)
    print(test_generation_results)

    print("Retain Set (shared dataset) Results:")
    print(retain_classification_results)
    print(retain_generation_results)

    print("Retain Set (real person) Results:")
    print(real_person_classification_results)
    print(real_person_generation_results)

    output_file = f'{args.output_folder}/{args.output_file}_evaluation_results.json'
    # Prepare the data to be saved in JSON format
    results_data = {
        "Forget Set Results": {
            "classification": forget_classification_results,
            "generation": forget_generation_results
        },
        "Test Set Results": {
            "classification": test_classification_results,
            "generation": test_generation_results
        },
        "Retain Set (shared dataset) Results": {
            "classification": retain_classification_results,
            "generation": retain_generation_results
        },
        "Retain Set (real person) Results": {
            "classification": real_person_classification_results,
            "generation": real_person_generation_results
        }
    }

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


