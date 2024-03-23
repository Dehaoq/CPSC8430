import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import Repository, get_full_repo_name, notebook_login
import evaluate
from tqdm.auto import tqdm
import numpy as np
import collections
import json


max_length = 256 
stride = 64
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
output_dir = "finetuned_bert"

def reformat_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    examples = []
    # iterate over 'data' list
    for elem in json_data['data']:
        title = elem['title']

        # iterate over paragraphs
        for paragraph in elem['paragraphs']:
            context = paragraph['context']

            # iterate over question-answers for this paragraph
            for qa in paragraph['qas']:
                example = {}
                example['id'] = qa['id']
                example['title'] = title.strip()
                example['context'] = context.strip()
                example['question'] = qa['question'].strip()
                example['answers'] = {}
                example['answers']['answer_start'] = [answer["answer_start"] for answer in qa['answers']]
                example['answers']['text'] = [answer["text"] for answer in qa['answers']]
                examples.append(example)
    
    out_dict = {'data': examples}

    output_json_file = 'out_'+json_file
    with open(output_json_file, 'w') as f:
        json.dump(out_dict, f)

    return output_json_file


def train_model(model, train_dataloader, eval_dataloader, epochs):
    training_steps = epochs * len(train_dataloader)

    accelerator = Accelerator(mixed_precision='fp16')
    optimizer = AdamW(model.parameters(), lr = 5e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )


    for epoch in range(epochs):
        # train for 1 epoch
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            

        # # evaluate after each epoch 
        # accelerator.print("Evaluation...")
        # metrics = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'], accelerator)
        # print(f"epoch {epoch}:", metrics)

        # save and upload 
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)


def evaluate_model(model, dataloader, dataset, dataset_before_preprocessing, accelerator=None):
    if not accelerator: 
        accelerator = Accelerator(mixed_precision='fp16')
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    
    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(dataloader):
        with torch.no_grad(): 
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(dataset)]
    end_logits = end_logits[: len(dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, dataset, dataset_before_preprocessing
    )
    return metrics



def preprocess_training_examples(examples):
    questions = [question.strip() for question in examples['question']]
    inputs = tokenizer(
        questions, 
        examples['context'],
        max_length = max_length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    offset_mapping = inputs.pop('offset_mapping')
    sample_map = inputs.pop('overflow_to_sample_mapping')
    answers = examples['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # find start and end of the context
        idx = 0
        while sequence_ids[idx] != 1: 
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if answer not fully inside context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs


def process_validation_examples(examples):
    questions = [question.strip() for question in examples['question']]
    inputs = tokenizer(
        questions, 
        examples['context'],
        max_length = max_length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    sample_map = inputs.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(inputs['input_ids'])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offsets = inputs['offset_mapping'][i]
        inputs["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None for k, offset in enumerate(offsets)
        ]

    inputs['example_id'] = example_ids
    return inputs




metric = evaluate.load("squad")

n_best = 20
max_answer_length = 30

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features): 
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        # loop thru all features associated with example ID
        for feature_index in example_to_features[example_id]: 
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes: 
                for end_index in end_indexes: 
                    # skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None: 
                        continue
                    # skip answers with a length that is either <0 or >max_answer_length
                    if end_index < start_index or end_index-start_index+1 > max_answer_length: 
                        continue
                    
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                    answers.append(answer)
        # select answer with best score
        if len(answers) > 0: 
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else: 
            predicted_answers.append({"id": example_id, "prediction_text": ""})
        
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)