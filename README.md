# AIObioEnts: All-in-one biomedical entities

Biomedical named-entity recognition following the all-in-one NER scheme. The scripts in this repository are taken from the original [AIONER repository][AIONER_repo], with slight modifications in order to make them able to run with the latest package versions, in addition to removing some options and accommodating additional pre-trained models. If you use these models, please cite both this and the original AIONER repository, plus the [AIONER paper](https://doi.org/10.1093/bioinformatics/btad310).

The data used for training is also provided with slight modifications. In particular, we found some overlaps between one of the test sets and the merged training set; in this repository, there are no overlaps.

**Table of contents**

- [A note on language](#a-note-on-language)
- [Core biomedical entities](#core-biomedical-entities)
- [Additional biomedical entities](#additional-biomedical-entities)
- [Model files](#model-files)
- [Usage](#usage)
- [Visualisation](#visualisation)
- [References](#references)
- [Changelog](#changelog)


## A note on language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Core biomedical entities

We have retrained the original implementation based on the BioRED dataset, along with additional BioRED-consistent datasets:
- Gene: GNormPlus, NLM-Gene, DrugProt
- Disease: BC5CDR, NCBI Disease
- Chemical: BC5CDR, NLM-Chem, DrugProt
- Species: Species-800, Linnaeus
- Variant: tmVar
- Cell line: BioID

using four pre-trained language models as a base:
- [BiomedBERT-base pre-trained on abstracts from PubMed; the best-performing model reported in the original AIONER paper](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
- [BiomedBERT-base pre-trained on both abstracts from PubMed and full-texts articles from PubMedCentral](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
- [BioLinkBERT-base](https://huggingface.co/michiyasunaga/BioLinkBERT-base)
- [BioLinkBERT large](https://huggingface.co/michiyasunaga/BioLinkBERT-large)


**F1 scores**

The F1 scores of the current implementation on the BioRED test set are shown below:

|               | **BiomedBERT-base abstract** | **BiomedBERT-base abstract+fulltext** | **BioLinkBERT-base** | **BioLinkBERT-large** |
| ------------- | :--------------------------: | :-----------------------------------: | :------------------: | :-------------------: |
| **Cell line** |            88.66             |                 92.93                 |        91.67         |         93.75         |
| **Chemical**  |            92.61             |                 93.04                 |        92.97         |         94.01         |
| **Disease**   |            88.89             |                 88.40                 |        88.17         |         88.90         |
| **Gene**      |            94.82             |                 94.59                 |        95.84         |         96.02         |
| **Species**   |            96.34             |                 96.72                 |        97.97         |         96.70         |
| **Variant**   |            93.81             |                 94.17                 |        93.72         |         95.00         |
| **Overall**   |          **92.28**           |               **92.68**               |      **93.14**       |       **93.69**       |


## Additional biomedical entities

We fine-tune the models using a modified version of the latest release of the [AnatEM](https://nactem.ac.uk/anatomytagger/#AnatEM) corpus, including only selected entities that are of interest to us: *cell component*, *tissue*, *organ*, *multi-tissue structure*, and *organ*, along with the newly-introduced *cancer*. 

**F1 scores**

The F1 scores for the 4 models on the test set of this modified dataset are shown below:

|                            | **BiomedBERT-base abstract** | **BiomedBERT-base abstract+fulltext** | **BioLink-base** | **BioLink-large** |
| -------------------------- | :--------------------------: | :-----------------------------------: | :--------------: | :---------------: |
| **Cell component**         |            85.00             |                 82.54                 |      76.72       |       84.25       |
| **Tissue**                 |            70.92             |                 70.82                 |      71.95       |       72.19       |
| **Cancer**                 |            87.36             |                 84.13                 |      88.29       |       86.56       |
| **Organ**                  |            74.74             |                 76.47                 |      77.01       |       81.94       |
| **Multi-tissue structure** |            67.87             |                 67.36                 |      72.77       |       77.96       |
| **Overall**                |          **79.29**           |               **77.86**               |    **80.60**     |     **81.30**     |


## Model files

All the 8 trained models can be downloaded from [HuggingFace](https://huggingface.co/datasets/SIRIS-Lab/AIObioEnts-model_files/). The pre-trained models can be obtained from the links above.

## Usage

All steps work exactly as with the original scripts, differing in two aspects:
- The `decoder` parameter has been removed and only CRF is used
- The type of model can be any of `pubmedbert`, `pubmedbert_full`, `biolink_base`, `biolink_large`

**Training example**

````bash
python AIONER_Training.py -t ../data/conll/ALL_TRAIN/ALL_TRAIN.conll -e pubmedbert -o ../models/pubmedbert/
````

**Fine-tuning example**

````bash
python AIONER_FineTune.py -t ../data/AnatEM/conll/AnaTEM_selected_train.conll -d ../data/AnatEM/conll/AnaTEM_selected_dev.conll -v ../vocab/AnatEM_selected_label.vocab -m pubmedbert -o ../models/AnatEM/pubmedbert/
````

**Inference example**

````bash
python AIONER_Run.py -i ../data/AnatEM/pubtator/ -m ../models/AnatEM/pubmedbert/pubmedbert-best-finetune.h5 -v ../vocab/AnatEM_selected_label.vocab -e ALL -o path_to_output_folder
````

Please refer to the [AIONER repository][AIONER_repo] for further details.

## Visualisation

We provide an [example notebook](./notebooks/visualise_examples.ipynb) that uses [spaCy](https://spacy.io/) to visualise model outputs on plain text using a sample from a collection of title+abstract from [500 recent PubMed publications](./data/PubMed_Jun24/PubMed_Jun24.csv).

![Visualisation of entities from a PubMed publication](./img/visualisation.png)

## References

[[1] Ling Luo, Chih-Hsuan Wei, Po-Ting Lai, Robert Leaman, Qingyu Chen, and Zhiyong Lu. "AIONER: All-in-one scheme-based biomedical named entity recognition using deep learning." Bioinformatics, Volume 39, Issue 5, May 2023, btad310.](https://doi.org/10.1093/bioinformatics/btad310)

[AIONER_repo]: https://github.com/ncbi/AIONER

## Changelog

All notable changes to this project will be documented in the 📝 [CHANGELOG](CHANGELOG.md) file, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
