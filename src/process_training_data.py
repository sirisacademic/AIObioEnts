# -*- coding: utf-8 -*-
import os
import json
import re
import spacy

nlp = spacy.load('en_core_web_lg')

def pubtator_to_sentences_jsonl(conversion_list, input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(conversion_list, 'r') as conversion_file:
        conversion = conversion_file.readlines()
    conversion_file.close()

    for line in conversion:
        filename = input_path+'/'+line.split('\t')[0]
        entity = line.split('\t')[1]
        outfile = output_path+'/{}_{}_sentences.jsonl'.format(line.split('\t')[0].split('.')[0], entity)
        
        if entity != 'ALL':
            convert = line.split('\t')[2]
    
        with open(filename, 'r') as infile:
            all_text = infile.read().strip().split('\n\n')
    
            if entity!='ALL':
                mapping_i = {convert.split(':')[0]: convert.split(':')[1].strip('\n').split('|')}
                mapping = {}
                for k,v in mapping_i.items():
                    for x in v:
                        mapping[x] = k
    
            new_lines = []
            for doc in all_text:
                lines = doc.split('\n')
                
                # Process the title
                seg = lines[0].split('|t|')
                pmid = seg[0]
                title = seg[1]
                intext_title = '<{}>{}</{}>'.format(entity, title, entity)
                
                # Set offset for title (length of <entity> tag)
                title_offset = len('<{}>'.format(entity))
                
                # Initialize annotation list for the title
                title_annotations = []
                if len(lines) > 2:
                    for annotation in lines[2:]:
                        annotation = annotation.split('\t')
                        start, end, entity_type = int(annotation[1]), int(annotation[2]), annotation[4]
                        if start < len(title):  # Check if the annotation is in the title
                            if entity != 'ALL' and entity_type in mapping:
                                title_annotations.append([start + title_offset, end + title_offset, mapping[entity_type]])
                            elif entity == 'ALL':
                                title_annotations.append([start + title_offset, end + title_offset, entity_type])
        
                # Add title to the output
                new_lines.append({'id': pmid, 'text': intext_title, 'label': title_annotations})
                
                # Process the abstract
                seg = lines[1].split('|a|')
                abstract = seg[1]
                
                # Assuming there exists a function to split abstract into sentences
                sentences = nlp(abstract)  # Split the abstract into sentences
                
                # Track offset for the current sentence within the abstract
                current_offset = len('{} '.format(title))  # Start after title + space
                
                # Process annotations and match them to sentences
                if len(lines) > 2:
                    for sentence_nlp in sentences.sents:
                        sentence = sentence_nlp.text
                        intext_sentence = '<{}>{}</{}>'.format(entity, sentence, entity)
                        sentence_annotations = []
        
                        sentence_start_offset = current_offset
                        sentence_end_offset = sentence_start_offset + len(sentence)
                        
                        for annotation in lines[2:]:
                            annotation = annotation.split('\t')
                            start, end, entity_name, entity_type = int(annotation[1]), int(annotation[2]), annotation[3], annotation[4]
                            
                            # Check if annotation falls within the current sentence
                            if sentence_start_offset <= start < sentence_end_offset:
                                adjusted_start = start - sentence_start_offset + len('<{}>'.format(entity))
                                adjusted_end = end - sentence_start_offset + len('<{}>'.format(entity))
                                
                                if entity != 'ALL' and entity_type in mapping:
                                    sentence_annotations.append([adjusted_start, adjusted_end, mapping[entity_type]])
                                elif entity == 'ALL':
                                    sentence_annotations.append([adjusted_start, adjusted_end, entity_type])
        
                        # Append the sentence to new_lines
                        new_lines.append({'id': pmid, 'text': intext_sentence, 'label': sentence_annotations})
                        
                        # Update the current offset for the next sentence
                        current_offset += len(sentence) + 1  # Adding 1 for space between sentences
    
        infile.close()
    
        with open(outfile, 'w') as fout:
            for ddict in new_lines:
                jout = json.dumps(ddict) + '\n'
                fout.write(jout)
        fout.close()


def formatted_annotation(text, labels):
  formatted_entities = []

  # Initialize the start index for the next labeled entity
  start_index = 0

  for start, end, entity_type in labels:
      # Append the non-labeled part of the text
      formatted_entities.append(text[start_index:start])

      # Append the labeled part in the specified format
      entity_text = text[start:end]
      formatted_entity = f'{{{entity_text}}}[{entity_type}]'
      formatted_entities.append(formatted_entity)

      # Update the start index for the next iteration
      start_index = end

  # Append any remaining non-labeled text
  formatted_entities.append(text[start_index:])

  formatted_text = ''.join(formatted_entities)
  return formatted_text
    

def preprocess_punctuation(raw_text: str, labels: list):
  punctuations = '\'!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~'
  
  for punct in list(punctuations):
    raw_text = raw_text.replace(punct, f' {punct} ')
  raw_text = re.sub(r'\s+', ' ', raw_text)
  for label in labels+['ALL']:
    raw_text = raw_text.replace(f'[ {label} ]', f'[{label}]')
    raw_text = raw_text.replace(f'< {label} >', f'<{label}>')
    raw_text = raw_text.replace(f'< / {label} >', f'</{label}>')
    raw_text = re.sub(r'\s*\}\s*', '}', raw_text).strip()
    raw_text = re.sub(r'\s*\{\s*', ' {', raw_text).strip()

  return raw_text
    

def get_tokens_with_entities(raw_text: str, labels: list):
    raw_text = preprocess_punctuation(raw_text, labels)
    entity_type = raw_text.split()[0][1:-1]
    raw_tokens = re.split(r"\s(?![^\{]*\})", raw_text)
    entity_value_pattern = r"\{(?P<value>.+?)\}\[(?P<entity>.+?)\]"
    entity_value_pattern_compiled = re.compile(entity_value_pattern, flags=re.I|re.M)
    tokens_with_entities = []
    for raw_token in raw_tokens:
        match = entity_value_pattern_compiled.match(raw_token)
        if match:
            raw_entity_name, raw_entity_value = match.group("entity"), match.group("value")
            for i, raw_entity_token in enumerate(re.split("\s", raw_entity_value)):
                entity_prefix = "B" if i == 0 else "I"
                entity_name = f"{entity_prefix}-{raw_entity_name}"
                tokens_with_entities.append((raw_entity_token, entity_name))
        else:
            if entity_type!='ALL':
                tokens_with_entities.append((raw_token, f'O-{entity_type}'))    
            else:
                tokens_with_entities.append((raw_token, "O"))
    return tokens_with_entities


class NERDataMaker:
    def __init__(self, texts, labels):
        self.unique_entities = []
        self.processed_texts = []
        temp_processed_texts = []
        for text in texts:
            tokens_with_entities = get_tokens_with_entities(text, labels)
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)
        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")
        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities])

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)
            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }
        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def as_hf_dataset(self, tokenizer):
        from datasets import Dataset, Features, Value, ClassLabel, Sequence
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        ids, ner_tags, tokens, strings = [], [], [], []

        for i, pt in enumerate(self.processed_texts):
            ids.append(i)
            pt_tokens,pt_tags = list(zip(*pt))
            ner_tags.append(pt_tags)
            tokens.append(pt_tokens)
            strings.append(' '.join(pt_tokens))

        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens,
            "strings": strings
        }
        features = Features({
            "strings": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities)),
            "id": Value("int32")
        })
        ds = Dataset.from_dict(data, features)
        tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
        return tokenized_ds