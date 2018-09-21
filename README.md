# Three-attention QANet with elmo
This is a tensorflow implementation of three attention with elmo model. The input embedding layer is similar to the one in the QANet model except for our adding elmo part. As for the embedding encoding layer, the same one in QANet is used. And we also have the similar layer like context-query attention layer. Candidate answers information are added. We calculated the bi-attention between query and context first and then calculate the bi-attention between candidate answers and query-context. Finally, we only have an output layer. After getting three-attention vectors, we multiply them by transformed candidates matrix. And corss entropy loss between predicted vector and ground true label vector is used. In the testing, we select the one which has maximum probability as the answer.

#![Alt text](/../master/screenshots/three_attention_with_elmo.png?raw=true "Network Outline")

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

## Prepare
(1) To download [Qangaroo Dataset](http://qangaroo.cs.ucl.ac.uk/index.html) first. Here we used unmask train and dev dataset files (qangaroo_v1.1/wikihop/train.json and qangaroo_v1.1/wikihop/dev.json) to train the model.

(2) To run filtering data python file (qangaroo2squad_preprocess.py). And It can also transfer qangaroo format to [squad 1.1](https://rajpurkar.github.io/SQuAD-explorer/) format dataset files. Here you need to set input file variables (train_in and dev_in) to two dataset files described in (1). And you need to set output file variables (train_out and dev_out) and then you can get two json filtered [squad 1.1](https://rajpurkar.github.io/SQuAD-explorer/) format dataset files. 

The filtering strategies for train and dev dataset files are as following:

Train: We only keep the sentences containing question and answer;

Dev: We keep the sentences containing question and answer candidates.

(3) To run "sh download.sh" to get Glove embedding representations.

(4) Here we will use elmo embedding to enhance the input word embeddings. And you need to download required files elmo will use. The required files are listed in [bilm-tf/usage_character.py](https://github.com/allenai/bilm-tf/blob/master/usage_character.py). [Bilm-tf](https://github.com/allenai/bilm-tf) show that tensorflow 1.2 is needed. Based on our experiments, tensorflow 1.5 is ok for [bilm-tf](https://github.com/allenai/bilm-tf).

P.S. "lm_weights.hdf5" file is needed to generate by yourself. About how to generate please refer to [bilm-tf](https://github.com/allenai/bilm-tf).

(5) To run the command "python config.py --mode prepro" to generate preprocessed dataset files. You need to specify the elmo required files which are listed in (4). In config.py, some parameters should be set based on your directories. And many hyper-parameters can be adjusted to get better performance. 

## Train
To run the command "python config.py --mode train" to train the model. During the training process, the classification accuracy in part training dataset (fixed 100 * batch_size training samples) and the whole dev dataset will be printed after every epoch.

P.S. Because the number of training samples is large (about 43, 000), calculating the performance in the whole training dataset after every epoch is slow.

## Test
To run the command "python config.py --mode train" to test the model. The classification accuracy in test dataset will be printed.


