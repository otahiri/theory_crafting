# Transformer
a Transformer is a neural network that work with the self attention mechanism,
it follow these stops [embedding](#embedding) -> [block](#block) -> [output](#output) the output will be dependent on the type of the transformer,
e.i encoder only decoder only or encoder and decoder

## Block
this block consist of 4 different layers the data it outputs will be fed to a next block total of n times depending on the type of transformer
[normalize](#normalize) -> [multi-headed-attentions](#multi-headed-attentions) -> [add](#add) ->[normalize](#normalize) -> [feed-forward](#feed-forward) -> [add](#add) 
## Embedding
Assuming we are past the training, there will be a matrix that has the shape of [dimension of the model / vocab size].
This matrix has the embedding of all tokens, so to get the contextualize embedding of the token, we simply take the scalar at the index of the token(an integer).
We then take the [positional embedding](#positional-embedding) of the token.
We then do element wise addition simply put we add element at x, y in contextualize embedding to the element at x, y in the positional embedding.
This give us a vector containing the context of the token and is spatially aware,
"in other words if you multiply it with another vector it kinda knows the distance between the 2 tokens idk i hate math".
The result from this is a matrix with dimension [query length in tokens / model dimension] each row contain the vector representing the token, contextually and positionally.
### Multi Headed Attentions
depending on how many heads we define, or the manufacturer of the model decide to use, that number is then used to divide the embedded matrix into that number let us assume we have 4 header,
so a matrix of shape [100 / 512] will give us 8 different matrices of shape [100 / 128].
This way we can apply self attention from 4 heads and later concatenate all 4 into a single one with shape[100 / 512].
However there is a good balance between too many heads and too little heads, this golden ration is that each head will have between 64 and 128 dimensions, each head will take a sequence and apply its own self attention to it.
So the way this works is we get the result of the [embedding](#embedding) and multiply it with the matrices representing the key query and the value.
We get 3 different matrices with shape of [model dimensions / len of query] we then divide each matrix on the head count,
And apply [self attention](#self-attention) to each part and concatenate them, the result we then compute the dot product of that new matrix with the weights matrix.
This output is the sent to the [add/normalize](#add/normalize) layer.
### Feed Forward
We calculate the dot product of the weight matrix with the input matrix, and add to it the bias vector, we then apply an activation function to the result,
which then we calculate the dot product of that result with a second weight matrix and add to it a second bias vector.
 

### Normalize
we take the original input and normalize it and then pass it to the next sub-layer
this is the way most llms work now  with a pre-normalizing layer, i still do not know why tho, since this is about post-ln and pre-ln needing a warmup learning rate stage so the learning rate does not explode????

### Add
the output of each sub-layer is added to the original input, so the input does not get squashed, since the main idea behind the addition layer is to keep track of the original values,

#### Self Attention
using the key query and value we then apply a simple formula, we multiply the query with the key of every other token we then get a single vector size of len of the query (the token count).
We then divide that whole vector with the square root of the dimension of the head if using multi headed attention or the dimensions of the model if using single headed self attention.
We will apply softmax to that vector so the total will add up to 1.0. This new vector will be multiplied by the value vector this will give us a vector.
In reality the model does this for all tokens at once, so in actual practice the steps are:
query matrix which is  [token len / head embedding dimensions] shaped will be be multiplied (dot product) with the key matrix which is the same shape this will give as a matrix,
it is with shape [token len / token len] this will be divided with the square root of the head dimensions, and softmaxed so the sum of every dimension will be 1.0.
This new matrix will be multiplied with the value matrix with shape [token len / head dimensions].
This result will shape of [token len /  head dimensions].


### Positional Embedding
The positional embedding of all tokens in the vocab is made during the boot up of the model.
This follows the same logic as the contextualize embedding, we simply make a matrix with the shape [dimension of the model / vocab size], and we take the scalar at the index of the token.
To get this matrix we use the [Sinusoidal Formula](#sinusoidal-formula) to get the positional embedding of all vectors

#### Sinusoidal Formula
the formula give us a relational value for each token so if we multiply the value of 2 tokens it give us the cosine of the difference between 2 tokens.
This follow the trigonometrical rule of cos(A)*cos(B) + sin(A)*sin(B) = cos(A - B), this entire thing is for the vectors to be spatially aware of where they are in relation to other vectors.
So we loop for i from 0 to dimension of model:
if i is even  we do PE(pos, 2i) = cos(pos / 10000*(2i / d-model))
if i is odd we do PE(pos, 2i + 1) = sin(pos / 10000*(2i / d-model))
where pos is the position of the token inside the  query, we do limit pos to context window, because the moment we surpass it the first token is forgotten.

##### Key
this simply represent the answer the query asks, so if a key is good enough for the query, it will result in a high match score(dot product between the key and query matrices will be high).

##### Query
the question the token asks it looks for the most appropriate key from all the previous tokens
##### Value
the value of the token itself
