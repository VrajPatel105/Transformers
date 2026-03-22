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
