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

The filtering straties for train and dev dataset files are as following:

Train: We only keep the sentences containing question and answer;

Dev: We keep the sentences containing question and answer candidates.

(3) Run the command "python config.py --mode prepro" to

```bash
# download SQuAD and Glove
sh download.sh
# preprocess the data
python config.py --mode prepro
```

Just like [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net), hyper parameters are stored in config.py. To debug/train/test/demo, run

```bash
python config.py --mode debug/train/test/demo
```

To evaluate the model with the official code, run
```bash
python evaluate-v1.1.py ~/data/squad/dev-v1.1.json train/{model_name}/answer/answer.json
```

The default directory for the tensorboard log file is `train/{model_name}/event`


## Detailed Implementaion

  * The model adopts character level convolution - max pooling - highway network for input representations similar to [this paper by Yoon Kim](https://arxiv.org/pdf/1508.06615.pdf).
  * The encoder consists of positional encoding - depthwise separable convolution - self attention - feed forward structure with layer norm in between.
  * Despite the original paper using 200, we observe that using a smaller character dimension leads to better generalization.
  * For regularization, a dropout of 0.1 is used every 2 sub-layers and 2 blocks.
  * Stochastic depth dropout is used to drop the residual connection with respect to increasing depth of the network as this model heavily relies on residual connections.
  * Query-to-Context attention is used along with Context-to-Query attention, which seems to improve the performance more than what the paper reported. This may be due to the lack of diversity in self attention due to 1 head (as opposed to 8 heads) which may have repetitive information that the query-to-context attention contains.
  * Learning rate increases from 0.0 to 0.001 in the first 1000 steps in inverse exponential scale and fixed to 0.001 from 1000 steps.
  * At inference, this model uses shadow variables maintained by the exponential moving average of all global variables.
  * This model uses a training / testing / preprocessing pipeline from [R-Net](https://github.com/HKUST-KnowComp/R-Net) for improved efficiency.


