# Multimodal Large Language Model Unlearning Benchmark (MLLMU-Bench)

## Benchmark Data Setup
Due to the size of the dataset, we place the dataset in the supplement file of the submission.
In particular, we have four datasets: `Full_Set.zip`, `Retain_Set.zip`, `Test_Set.zip`, and `Train_data.zip`.
Here are the explanations of each zip:
- `Full_Set.zip`: The full dataset that contains all the data for the benchmark. It contains 500 fictitious images
and 500 fictitious profiles corresponding to each image. In particular:
  - `classification_qa`: classification questions and answers corresponding to each profile.
  - `data_original`: the original image corresponding to each profile.
  - `data_profile`: the detailed profile information of each fictitious profile.
  - `classification_qa`: the designed classification questions of each person.
  - `generation_qa`: the designed generation questions of each person.
- Both `Retain_Set.zip` and `Test_Set.zip` contain the same data structure as `Full_Set.zip`. The only difference is that the
data original in Test Set contained the transformed images using latent diffusion model or random noise. 
- `Train_data.zip`: The training data for the benchmark. For each person, we created questions of each attributes in the format of 
question and answer. The training data is used to train the vanilla model of the benchmark.
- `Data_split`: The data split folder that contains the split of the data for the benchmark. It contains the split of the forget/retain data
based on the forget ratio (from the paper, we have 5%, 10% and 15%).
### Data Creation

## Fine-tune process
```finetune.py``` is the fine-tune file of how to fine-tune the multimodal large language model on the benchmark dataset (Train data).
Just run the following command:
```angular2html
python finetune.py --model_id [MODEL_ID] --save_dir [Save_DIR] --batch_size [batch_size] --lr [learning rate] --num_epochs [epoch num] --max_length 384
```
Make sure you have the Train_data.zip unzipped and place it in the same folder as the finetune.py file.

## Evaluation process
```eval.py``` is the evaluation file of how to evaluate the multimodal large language model on the benchmark dataset in both multimodal and unimodal evaluation,
both in classification and generation tasks.
Here is how to evaluate the saved model:
```angular2html
python eval.py \
 --model_id [MODEL_ID] \
 --cache_path [cache_path] \
 --test_image_folder ../Test_Set/data_original \
 --forget_image_folder ../Full_Set/data_original \
 --celebrity_classification_question_folder ../Retain_Set/classification_qa \
 --celebrity_generation_question_folder ../Retain_Set/generation_qa \
 --celebrity_image_folder ../Retain_Set/data_original \
 --output_file [output_file] \
 --output_folder [output_folder] \
 --forget_ratio 5 \
 --data_split_folder ../Data_split
```

## Baseline running
Here, we also provided unlearning baselines that we used in the benchmark. To run unlearning process
on those baselines, you need to first obtain the vanilla model and run the baseline file. Here, we use one
```GA.py``` file as an example, here is how to run GA file:
```angular2html
python GA.py \
--model_id [MODEL_ID] \
--vanilla_dir [Vanilla_dir] \
--save_dir [saved_dir] \
--batch_size [batch_size] \
--lr [lr] \
--num_epochs [num_epochs] \
--max_length 384 \
--data_split_dir ../Data_split \
--forget_split_ratio 5
```
