# Transformer vs Sentence Transformer
The main difference between a transformer (decoder only) and sentence transformer is how they deal with the output layer, we pass the query into the shared [steps](transformer) between the 2.
The output is the thing where they deal with it differently.
The [Sentence Transformer](#sentence-transfomer) need to output a percentage for the next token.
The [Transformer](#transformer) need to give a vector that contains the meaning of the whole query.

## Sentence Transformer
For the sentence transformer  the output of the shared steps is passed into a pooling step.
There are many pooling methods like average pooling max pooling sum pooling and last token pooling. the others do not matter since they dominate other fields.
What we care about and the one that is used in sentence transformers is [average/mean pooling](#average/mean-pooling)


#### Average/Mean Pooling
the average pooling simply put takes the average of all the vectors in the model dimensions axis and sum them up and divide them by the length for that vector.
This however will introduce  a new issue which is the size of each sentence is bound to be unique, some are 5 tokens long some are thousands of tokens long, to fix this issue we use padding.
Basically we take the longest vector and use it as a basis for the rest, we pad the rest with [PAD] token, this token is then assigned the value of 0 and is used only to make the vectors similar.
When we divide by the length we do not count the padding as it will mess up with the average. This will give us a final vector that has the context from all the other tokens


## Transformer
to my knowledge the output from the shared steps is multiplied by a single unembedding vector and then the output is softmaxed so the token with the highest value is the next token.
