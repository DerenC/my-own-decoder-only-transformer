## Iteration versions' description

> Keep track of lowest training loss & validation loss, and generated output (1000 tokens) for each iteration

1. [DONE] Basic Bigram Language Model + Character level tokeniser
    - Predict the next token by using only the last token before it.
2. [DONE] Previous tokens as Bag-of-words (BOW) all have equal weights
    - Every token in the context that is before the target token to be predicted, has equal attention weights.
3. [DONE] Add single head self-attention mechanism
4. [DONE] Include encoded position embedding
5. [DONE] Add multiple heads of self-attention & concatenate the results
6. [DONE] Add feed forward layer
7. [DONE] Sequentially repeat a few blocks containing a multi-head attention layer and a feed forward layer
    - This is when the neural network gets too deep and suffers from optimization issue.
8. [DONE] 1st Optimization: Add residual connection
9. [DONE] 2nd Optimization: Add layer normalizations before inputting to modules (pre-norm formulation)
10. Add dropout as a regularization technique before scaling up the model
11. Scale up the model by increasing the relevant hyperparamaters
12. Use GELU instead of ReLU (may not matter as it may just be for loading checkpoints)

Other variables:
- Optimizer used (AdamW & SGD)
- Tokenizer used (BPE, WordPiece, Word-level)
- Pre & post layer normalization (before and after the modules)
- Concatenate outputs of multiple heads VS Treating heads as a separate dimension (may not have any effect other than faster training time)