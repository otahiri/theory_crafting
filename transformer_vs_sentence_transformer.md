# Transformer vs Sentence Transformer
The main difference between a transformer (decoder only) and sentence transformer is how they deal with the output layer, we pass the query into the shared [steps](transformer) between the 2.
The output is the thing where they deal with it differently.
The [Transformer](#transformer) need to output a percentage for the next token.
The [Sentence Transformer](#sentence-transfomer) need to give a vector that contains the meaning of the whole query.
## Difference
So the mains difference between the 2 is as follows:
### Attention Step
#### the sentence transformer
Does not edit the output from the self attention step 
#### the transformer
Takes the result of the self attentions step and apply a triangular mask thereby the resulting matrix will allow each token to look at the values of the previous tokens and itself.

### the output step
#### the sentence transformer
Apply a [mean/average pooling](#average/mean-pooling) to the result of the transformer block.
#### the transformer
Takes the vector representing the last token from the result of the transformer block, dot product it with the transpose of a hidden wight matrix of shape [vocab size / model dimensions].
The resulting vector is then softmaxed. We take the index of the max arg of the vector as our next token.



## Sentence Transformer
For the sentence transformer  the output of the transformer block is passed into a pooling step.
There are many pooling methods like average pooling max pooling sum pooling and last token pooling. the others do not matter since they dominate other fields.
What we care about and the one that is used in sentence transformers is [average/mean pooling](#average/mean-pooling)


#### Average/Mean Pooling
The average pooling simply put takes the average of all the vectors in the model dimensions axis and sum them up and divide them by the length for that vector.
This however will introduce  a new issue which is the size of each sentence is bound to be unique, some are 5 tokens long some are thousands of tokens long, to fix this issue we use padding.
Basically we take the longest vector and use it as a basis for the rest, we pad the rest with [PAD] token, this token is then assigned the value of 0 and is used only to make the vectors similar.
When we divide by the length we do not count the padding as it will mess up with the average. This will give us a final vector that has the context from all the other tokens


## Transformer
We take the last layer of the result of the transformer block is  i.e vector representing the last token of the sequence.
This is then dot-producted with a hidden weight matrix of shape [vocab_size / model dimension]. We take the resulting vector and soft max it and we get a probability vector where the index of the max value is the next token.
