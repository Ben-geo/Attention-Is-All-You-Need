# Attention-Is-All-You-Need

This is a reproduction of 2017 paper attention is all you need

Let's start with the need for Transformer; a model that only used the attention and removed the need for recurrance or convultions for sequential data

### Limitations of Recurrance and Convulutions
#### Recurrance
**Lack of Parallelizaion :** RNN passes sequences back into the model. so for each hidden state you need to have alraedy computed the hidden state in the for the previous step. So inherently we can not parallelize them, This slows down training by a lot.

**Vanishing/Exploding Gradients :** Information is lost as more and more information is passed through the RNN. ( LSTM solves this to some extend but not entirely)

#### Convulutions
**Small Receptive Field :** Local blocks get access to data within that block alone (example top left and bottom right corner of an image) so to get information from both these points a lot of layers have to be stacked.

**Positional Invariance :** They are not good at capturing where the pattern is (a cats nose is a cats nose no matter where it is in the image). in sequences the position matters a lot, to solve this positional encoding can be done but convulutions itself doesn't prioritize position

