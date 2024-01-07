Anisotropic Span Embeddings and the Negative Impact of Higher-Order Inference for Coreference Resolution
========

An implementation of ACL 2021 submission: \
Anisotropic Span Embeddings and the Negative Impact of Higher-Order Inference for Corefernce Resolution.\
Our code is based on the [Xu and Choi, 2020](https://github.com/emorynlp/coref-hoi) and [Joshi et al. 2020](https://github.com/mandarjoshi90/coref). 

Code for measuring the degree of **anisotropy** is here: [Ethayarajh,2019](https://github.com/kawine/contextual)


**Files**:
* [run_scr.py](run_scr.py): training and evaluation
* [share_encoder_model.py](share_encoder_model.py): the coreference model that uses shared encoder for mention proposal and mention linking
* [data_aug.py](data_aug.py): generate synthesized documents
* [preprocess.py](preprocess.py): converting CoNLL files to examples
* [tensorize.py](tensorize.py): tensorizing example
* [experiments.conf](experiments.conf): different model configurations
* [higher_order.py](higher_order.py): higher-order inference modules

**Directories**:
* `datasets`: directory for datasets used for cased pre-trained models, such as SpanBERT
* `datasets_uncased`: directory for datasets used for uncased pre-trained models, such as ELECTRA
* `conll-2012`: directory for CoNLL-2012 coreference scorer


## Basic Setup
Set up environment and data for training and evaluation:
* Install Python3 dependencies: `pip install -r requirements.txt`
* Build original dataset (requiring [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus): `./setup_data.sh /path/to/ontonotes /path/to/data_dir`, 
  pay attention to the tokenizer for preprocessing data, use `bert-base-uncased` in `preprocess.py` when you are generating data for ELECTRA
* Generate **synthesized** documents: `python data_aug.py [config-name]`
* For **GAP** dataset, download it from [here](https://research.google/tools/datasets/gap-coreference/) and transform it using [Joshi et al. 2020](https://github.com/mandarjoshi90/coref) code. The scorer is from [Webster et al., 2018](https://github.com/google-research-datasets/gap-coreference)

## Training
`python run_scr.py [config] [gpu_id]`

* [config] can be any **configuration** in [experiments.conf](experiments.conf)
* Log file will be saved at `your_data_dir/[config]/log_XXX.txt`
* Models will be saved at `your_data_dir/[config]/model_XXX.bin`

## Hyperparameters
Some important hyperparameters in [experiments.conf](experiments.conf):
* `data_dir`: the full path to the directory containing dataset, models, log files
* `alpha`: the weight for layer 1 embeddings of contextualized representations, such as SpanBERT and ELECTRA
* `bert_pretrained_name_or_path`: the name of the pretrained [HuggingFace models](https://huggingface.co/models)
* `max_training_sentences`: the maximum segments to use when document is too long. All synthesized documents have 2 segments.

