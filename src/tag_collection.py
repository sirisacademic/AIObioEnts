# -*- coding: utf-8 -*-
import argparse
from tagging_fn import tag_pubtator, tag_json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='tag text collection, python tag_collection.py -t collectionType -i inputFile -o outputFile -m trainedModel -e entityType')
    parser.add_argument('--type', '-t', help='collection type', default='pubtator')
    parser.add_argument('--infile', '-i', help='input file')
    parser.add_argument('--outfile', '-o', help='output file')
    parser.add_argument('--model', '-m', help='path to trained model')
    parser.add_argument('--entity', '-e', help='entity type (Gene, Chemical, Disease, Variant, Species, CellLine, ALL)', default='ALL')
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    infile = args.infile
    outfile = args.outfile
    entity_type = args.entity
    file_type = args.type

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, add_prefix_space=True, trim_offsets=True, model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=0) # pass device=0 if using gpu

    if file_type == 'pubtator':
        tag_pubtator(infile, outfile, pipe, entity_type=entity_type)
    elif file_type == 'json':
        tag_json(infile, outfile, pipe, entity_type=entity_type)
    else:
        print('text collection file type not supported')