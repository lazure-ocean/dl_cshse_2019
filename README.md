# dl_cshse_2019
PyTorch implementation of a research paper [MaskGAN: Better Text Generation via Filling in the______](https://arxiv.org/abs/1801.07736) within the Deep Learning course at the HSE.

[Tensorflow implementation](https://github.com/tensorflow/models/tree/master/research/maskgan)

* MaskMLE: dl_cshse_2019/seq2seq/maskmle.py

* Algorithm:
1. Pretrain rnn cells (lstm) on language modelling task. 
2. Use the weights in generator and discrimantor encoders and decoders
3. Pretrain generator and discriminator in MaskMLE mode
4. Train generator and discriminator in GAN actor-crititc mode with following loop:
    * Fix discriminator parameters, train generator in actor-critic mode
    * Fix generator parameters, sample real and fake sentences, train discriminator
