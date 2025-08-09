# Attention-Is-All-You-Need

This is a reproduction of 2017 paper attention is all you need

Let's start with the need for Transformer; a model that only used the attention and removed the need for Recurrence or convultions for sequential data

## Limitations of Recurrence and Convolutions

### Recurrence

- **Lack of Parallelization :** RNN passes sequences back into the model. so for each hidden state you need to have already computed the hidden state in the the previous step. So inherently we can not parallelize them, This slows down training by a lot.
- **Vanishing/Exploding Gradients :** Information is lost as more and more information is passed through the RNN. ( LSTM solves this to some extent but not entirely)

### Convolutions

- **Small Receptive Field :** Local blocks get access to data within that block alone (example top left and bottom right corner of an image) so to get information from both these points a lot of layers have to be stacked.
- **Positional Invariance :** They are not good at capturing where the pattern is (a cats nose is a cats nose no matter where it is in the image). in sequences the position matters a lot, to solve this positional encoding can be done but Convolutions itself doesn't prioritize position

### Model Architecture

![Directly taken from the paper](model_architecture.png)


"the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence
of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output
sequence (y1, ..., ym) of symbols one element at a time."

#### Encoder

#### Decoder