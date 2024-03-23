import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, default_data_collator
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import collections
import json
from utils import *

if __name__ == "__main__":

    model_checkpoint = "bert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # SpokenSQuAD dataset files
    spoken_train = 'spoken_train-v1.1.json'
    spoken_test = 'spoken_test-v1.1.json'
    spoken_test_WER44 = 'spoken_test-v1.1_WER44.json'
    spoken_test_WER54 = 'spoken_test-v1.1_WER54.json'

    spoken_train = reformat_json(spoken_train)
    spoken_test = reformat_json(spoken_test)
    spoken_test_WER44 = reformat_json(spoken_test_WER44)
    spoken_test_WER54 = reformat_json(spoken_test_WER54)
    spoken_squad_dataset = load_dataset('json',data_files= { 'train': spoken_train,'validation': spoken_test,'test_WER44': spoken_test_WER44,'test_WER54': spoken_test_WER54 }, field = 'data')
    # Preprocessing training and testing data
    train_dataset = spoken_squad_dataset['train'].map(preprocess_training_examples,batched = True,remove_columns=spoken_squad_dataset['train'].column_names)
    validation_dataset = spoken_squad_dataset['validation'].map(process_validation_examples,batched = True,remove_columns=spoken_squad_dataset['validation'].column_names)
    test_WER44_dataset = spoken_squad_dataset['test_WER44'].map(process_validation_examples,batched = True,remove_columns=spoken_squad_dataset['test_WER44'].column_names)
    test_WER54_dataset = spoken_squad_dataset['test_WER54'].map(process_validation_examples,batched = True,remove_columns=spoken_squad_dataset['test_WER54'].column_names)

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    test_WER44_set = test_WER44_dataset.remove_columns(["example_id", "offset_mapping"])
    test_WER44_set.set_format("torch")
    test_WER54_set = test_WER54_dataset.remove_columns(["example_id", "offset_mapping"])
    
    test_WER54_set.set_format("torch")
    # Dataloader define
    train_dataloader = DataLoader(train_dataset, shuffle = True, collate_fn=default_data_collator, batch_size=8)
    eval_dataloader = DataLoader(validation_set, collate_fn=default_data_collator, batch_size=8)
    test_WER44_dataloader = DataLoader(test_WER44_set, collate_fn=default_data_collator, batch_size=8)
    test_WER54_dataloader = DataLoader(test_WER54_set, collate_fn=default_data_collator, batch_size=8)

    train_model(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, epochs = 1)
    test_metrics = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'])
    test_v1_metrics = evaluate_model(model, test_WER44_dataloader, test_WER44_dataset, spoken_squad_dataset['test_WER44'])
    test_v2_metrics = evaluate_model(model, test_WER54_dataloader, test_WER54_dataset, spoken_squad_dataset['test_WER54'])
    print("Test Valid Set, F1 : " + str(test_metrics['f1']))
    print("Test WER44    , F1 : " + str(test_v1_metrics['f1']))
    print("Test WER54    , F1 : " + str(test_v2_metrics['f1']))