# Three-attention QANet with elmo
This is a tensorflow implementation of three attention with elmo model. QANet paper is rank

![Alt text](/../master/screenshots/three_attention_with_elmo.png?raw=true "Network Outline")

## Dataset
The dataset used for this task is [Qangaroo Dataset](http://qangaroo.cs.ucl.ac.uk/index.html).
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

## Requirements
  * Python=3.5
  * NumPy
  * tqdm
  * TensorFlow=1.5
  * spacy==2.0.9 (only if you want to load the [pretrained model](https://drive.google.com/open?id=1gJtcPBNuDr9_2LuP_4x_4VN6_5fQCdfB), otherwise lower versions are fine)
  * bottle (only for demo)

## Preprocessing
(1) To download [Qangaroo Dataset](http://qangaroo.cs.ucl.ac.uk/index.html) first. After decompression, we use unmask train and dev dataset files (qangaroo_v1.1/wikihop/train.json and qangaroo_v1.1/wikihop/dev.json).

(2) To run filtering data python file (qangaroo2squad_preprocess.py). And It can also transfer qangaroo format to [squad 1.1](https://rajpurkar.github.io/SQuAD-explorer/) format dataset files. Here you need to set input file variables (train_in and dev_in) to two dataset files described in (1). And you need to set output file variables (train_out and dev_out) and then you can get two json filtered [squad 1.1](https://rajpurkar.github.io/SQuAD-explorer/) format dataset files. 

The filtering strategies for train and dev dataset files are as following:

Train: We only keep the sentences containing question and answer;

Dev: We keep the sentences containing question and answer candidates.

(3) To run "sh download.sh" to get Glove embedding representations.

(4) Here we will use elmo embedding to enhance the input word embeddings. And you need to download required files elmo will use. The required files are listed in [bilm-tf/usage_character.py](https://github.com/allenai/bilm-tf/blob/master/usage_character.py). [Bilm-tf](https://github.com/allenai/bilm-tf) show that tensorflow 1.2 is needed. Based on our experiments, tensorflow 1.5 is ok for [bilm-tf](https://github.com/allenai/bilm-tf).

P.S. "lm_weights.hdf5" file is needed to generate by yourself. About how to generate please refer to [bilm-tf](https://github.com/allenai/bilm-tf).

(5) Run the command "python config.py --mode prepro" to generate preprocessed dataset files. You need to specify the elmo required files which are listed in (4). In config.py, parameters should be set according to your machine.





