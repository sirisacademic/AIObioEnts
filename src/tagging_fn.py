# -*- coding: utf-8 -*-

import pandas as pd
from tqdm.autonotebook import tqdm
import spacy

tqdm.pandas()

nlp = spacy.load('en_core_web_lg')

def post_process_entities(text, entities, pipeline):
    results = []

    for entity in entities:
        if entity['entity'].startswith('O') or entity['word'].startswith('##'):
            continue
        results.append(entity)

    if results:
        for entity in results:
            entity['word'] = complete_subwords(entity['word'], entity['index']-1, pipeline.tokenizer.tokenize(text))
            entity['end'] = entity['start'] + len(entity['word'])
        return pipeline.group_entities(results)
    return []


def process_one_sentence(sentence, pipeline, entity_type='ALL'):
    entity_tag = '<{}>'.format(entity_type)
    closing_tag = '</{}>'.format(entity_type)
    offset = len(entity_tag)
    
    results = pipeline(entity_tag + sentence + closing_tag)

    index_offset = min([i for i, x in enumerate(pipeline.tokenizer.tokenize(entity_tag + sentence + closing_tag)) if x=='>'])

    for token in results:
        token['start'] -= offset
        token['end'] -= offset

        token['index'] -= index_offset

    return post_process_entities(sentence, results, pipeline)
    

def process_one_text(text, pipeline, entity_type='ALL'):
    doc = nlp(text)
    results = []
    offset = 0
    for sentence in doc.sents:
        to_tag = sentence.text
        result = process_one_sentence(to_tag, pipeline, entity_type)

        if offset>0:
            for entity in result:
                entity['start'] += offset
                entity['end'] += offset

        results.extend(result)
        
        offset += 1
        offset += len(to_tag)

    for entity in results:
        entity['word'] = text[entity['start']: entity['end']]

    return results


def complete_subwords(word, word_index, token_list):
    output_word = word
    idx = word_index
    while idx<len(token_list):
        if token_list[idx].startswith('##'):
            output_word+=token_list[idx][2:]
            idx+=1
        else:
            break
    
    return output_word


def tag_pubtator(input_file, output_file, pipeline, entity_type='ALL'):

    with open(input_file, 'r',encoding='utf-8') as fin:
        with open(output_file,'w', encoding='utf8') as fout:
            
            all_text = fin.read().strip().split('\n\n')
            
            for text in tqdm(all_text):
                
                lines = text.split('\n')
                seg = lines[0].split('|t|')
                text_id = seg[0]
                title = seg[1]
                
                seg = lines[1].split('|a|')
                abstract = seg[1]
                
                text_to_tag = title + ' ' + abstract

                

                entities = process_one_text(text_to_tag, pipeline, entity_type)

                fout.write(lines[0]+'\n'+lines[1]+'\n')
                
                for entity in entities:
                    fout.write('{}\t{}\t{}\t{}\t{}\n'.format(text_id, entity['start'], entity['end'], entity['word'], entity['entity_group']))
                fout.write('\n')

        fout.close()
    fin.close()

    return

    
def list_labels(text, pipeline, entity_type='ALL'):
    entities = process_one_text(text, pipeline, entity_type)

    results = []
    for entity in entities:
        results.append([entity['start'], entity['end'], entity['entity_group']])
    return results
    

def tag_json(input_file, output_file, pipeline, entity_type='ALL'):

    df = pd.read_json(input_file, lines=True)

    id, text = df.columns

    df['label'] = df[text].progress_apply(lambda x: list_labels(x, pipeline, entity_type))

    df.to_json(output_file, orient='records', lines=True)

    return