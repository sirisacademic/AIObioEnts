# -*- coding: utf-8 -*-
import argparse
import pandas as pd, numpy as np
from glob import glob

import process_training_data as pr
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import DatasetDict

import evaluate

seqeval = evaluate.load("seqeval")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='train AIONER model, python train_model.py -m pretrainedModel -d dataDir -v vocab -o outputDir -l learningRate -n nEpochs')
    parser.add_argument('--model', '-m', help='path to HF pretrained model')
    parser.add_argument('--data', '-d', help='directory of training JSONL files')
    parser.add_argument('--vocab', '-v', help='vocab file with BIO labels')
    parser.add_argument('--output', '-o', help='output directory of the trained model')
    parser.add_argument('--lr', '-l', help='learning rate', default=1e-5)
    parser.add_argument('--nepochs', '-n', help='number of training epochs', default=50)
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    DATA_DIR = args.data
    vocab = args.vocab
    OUTPUT_DIR = args.output
    lr = float(args.lr)
    nepochs = int(args.nepochs)

    train_files = glob(DATA_DIR+'/*.jsonl')

    data_raw = pd.concat([pd.read_json(file, lines=True) for file in train_files], ignore_index=True)
    data_raw.text = data_raw.text.astype('str')

    with open(vocab, 'r') as file:
        label_vocab = file.readlines()

    file.close()
    labels = [x.split('-')[1].strip() for x in label_vocab if x[0]=='B']

    print('\n========================================\npreprocessing training data...\n========================================\n')
    
    data_raw['text_annotated'] = data_raw.apply(lambda row: pr.formatted_annotation(row['text'],row['label']),axis=1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, add_prefix_space=True, trim_offsets=True, model_max_length=512)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    annotated_texts = data_raw['text_annotated'].tolist()

    print('\n========================================\npreparing dataset for training...\n========================================\n')

    dm = pr.NERDataMaker(annotated_texts, labels)
    
    dataset = dm.as_hf_dataset(tokenizer=tokenizer)

    # 80% train, 20% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'val': test_valid['train']})

    label_list_train = train_test_valid_dataset["train"].features[f"ner_tags"].feature.names
    label_list_val = train_test_valid_dataset["val"].features[f"ner_tags"].feature.names
    label_list_test = train_test_valid_dataset["test"].features[f"ner_tags"].feature.names

    label_list = label_list_train

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            }

    print('\n========================================\nloading pretrained model...\n========================================\n')

    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH,
                                                            ignore_mismatched_sizes=True,
                                                            num_labels=len(dm.unique_entities),
                                                            id2label=dm.id2label,
                                                            label2id=dm.label2id)

    training_args = TrainingArguments(output_dir='{}/chkpts'.format(OUTPUT_DIR),
                                      eval_strategy='epoch',
                                      learning_rate=lr,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      num_train_epochs=nepochs,
                                      weight_decay=0.01,
                                      save_strategy='epoch',
                                      load_best_model_at_end=True,
                                      metric_for_best_model='f1'
                                     )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_test_valid_dataset['train'],
                      eval_dataset=train_test_valid_dataset['val'],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      data_collator=data_collator
                     )

    print('\n========================================\ntraining model...\n========================================\n')

    trainer.train()

    print('\n========================================\nsaving model...\n========================================\n')

    trainer.save_model('{}/trained_model'.format(OUTPUT_DIR))


    print('\n========================================\nholdout set performance\n========================================\n')

    predictions = trainer.predict(train_test_valid_dataset['test'])

    all_predictions = []
    all_labels = []
    
    labels = train_test_valid_dataset['test']['labels']
    predictions = np.argmax(predictions.predictions, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(dm.id2label[predicted_idx])
            all_labels.append(dm.id2label[label_idx])
    
    print(seqeval.compute(predictions=[all_predictions], references=[all_labels]))