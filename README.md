# Bi-directional RNN Language Model in TensorFlow
Tensorflow implementation of Bi-directional RNN Langauge Model refer to paper [[Contextual Bidirectional Long Short-Term Memory Recurrent Neural Network Language Models: A Generative Approach to Sentiment Analysis]](http://www.aclweb.org/anthology/E17-1096).

<img src="https://user-images.githubusercontent.com/6512394/43193240-723549be-903a-11e8-8f3e-41500fdd156e.PNG">

## Requirements
- Python 3
- TensorFlow

## Usage
Penn Tree Bank (PTB) dataset is used for training and test. [ptb_data](ptb_data/) is copied from data/ directory of the [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz). 

### Train
```
$ python train.py
```

#### Hyperparameters
```
$ python train.py -h
usage: train.py [-h] [--model MODEL] [--embedding_size EMBEDDING_SIZE]
                [--num_layers NUM_LAYERS] [--num_hidden NUM_HIDDEN]
                [--keep_prob KEEP_PROB] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         rnn | birnn
  --embedding_size EMBEDDING_SIZE
                        embedding size.
  --num_layers NUM_LAYERS
                        RNN network depth.
  --num_hidden NUM_HIDDEN
                        RNN network size.
  --keep_prob KEEP_PROB
                        dropout keep prob.
  --learning_rate LEARNING_RATE
                        learning rate.
  --batch_size BATCH_SIZE
                        batch size.
  --num_epochs NUM_EPOCHS
                        number of epochs.
```


## Experimental Results
- Orange Line: LSTM language model
- Blue Line: Bi-directional LSTM language model

### Training Loss
<img src="https://user-images.githubusercontent.com/6512394/43238020-4f671d62-90c7-11e8-8086-a3ccca6548fd.PNG">

### Loss for Test Data
<img src="https://user-images.githubusercontent.com/6512394/43238017-4e2d1082-90c7-11e8-9f46-7699766db0bb.PNG">

## References
- [Contextual Bidirectional Long Short-Term Memory Recurrent Neural Network Language Models: A Generative Approach to Sentiment Analysis](http://www.aclweb.org/anthology/E17-1096)
- [https://medium.com/@plusepsilon/the-bidirectional-language-model-1f3961d1fb27](https://medium.com/@plusepsilon/the-bidirectional-language-model-1f3961d1fb27)
