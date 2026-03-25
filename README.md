# Transformers

Topics Covered:
### Covering the Pre-requisites
1. Encoder Decoder
   - Architecture of Encoder and Decoders
   - Encoder Forward Pass
   - Decoder Forward Pass
   - Improvements to make in very basic encoder decoder architecture
     - 1. using embeddings
       2. deep lstm's
       3. reversing the input
2. Attention Mechanism
   - Stating the main Problems with Vanilla Encoder Decoder Architecture
   - Deep dive on attention mechanism
   - Bahdanau Attention VS Luong Attention

### Transformers
 The main research paper used for this : [https://arxiv.org/pdf/1706.03762]

1. Self Attention
   - Introduction to Self Attention
   - Explaining the need for self attention
   - converting the embeddings to context aware embeddings
   - Query - Key - Value concept
   - Parallel Operations
   - Explaination behind Scaled Dot Product
   - Gemoetric Intuition
   - Why named 'Self Attention'
   - The problem with Self Attention - Why do we need multi head attention
2. Multi Head Attention
   - In depth explaination with example
   - Matrix Calculation
   - Problem with Self Attention - in context of sequence
3. Positional Encoding
   - Gradually explaining positional encoding concept from scratch using examples
   - getting positional encoding for a vector solved
   - THe Linear relationship property
4. Layer Normalization
   - Normalization
     - Where to apply?
     - Benefits of normalization
     - Revisiting Batch Normalization
     - Reasoning behind why not using batch normalization
     - Explaining Layer Normalization
5. Transformer Architecture
   - Encoder
     - Explaining all the parts step by step
       - inputs -> tokens -> embeddings -> positional encoding -> multi head attention -> add and normalization -> feed forward network -> add and normalization -> final output
       - repeat from multi head attentino to add to final output 6 times -> final output
   - Masked Self Attention
     - During Training
       - Sequential (time series)
       - Parallel
     - During Inference
   - Cross Attention
   - Transformer Decoder Architecture while Training -> Non-AutoRegressive
   - Transformer Decoder ARchitecture while Inference -> AutoRegressive
<img width="1920" height="16000" alt="erteryeryew" src="https://github.com/user-attachments/assets/6240dc62-c77e-45af-ad7c-3200e3afbcf3" />
