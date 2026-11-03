# Transformer
```
a Transformer is a neural network that work with the self attention mechanism, the output is a logits vector that has the shape of [dimension of the model / vocab size], the neural network follow these simple steps: 
[embedding](#embedding) -> [multi-headed-attentions](#multi-headed-attentions) -> [add/normalize](#add/normalize) -> [feed-forward](#feed-forward) -> [output](#output)
```

## Embedding
```
Assuming we are past the training, there will be a matrix that has the shape of [dimension of the model / vocab size].
This matrix has the embedding of all tokens, so to get the contextualize embedding of the token, we simply take the scalar at the index of the token(an integer).
we then take the [positional embedding](#positional-embedding) of the token.
we then do element wise addition simply put we add element at x, y in contextualize embedding to the element at x, y in the positional embedding.
this give us a vector containing the context of the token and is spatially aware,
"in other words if you multiply it with another vector it kinda knows the distance between the 2 tokens idk i hate math".
the result from this is a matrix with dimension [query length in tokens / model dimension] each row contain the vector representing the token, contextually and positionally.
```
## Multi Headed Attentions
depending on how many heads we define, or the manufacturer of the model decide to use, that number is then used to divide the embedded matrix into that number let us assume we have 4 header,
so a matrix of shape [100 / 512] will give us 8 different matrices of shape [100 / 128]. this way we can apply self attention from 4 heads and later concatenate all 4 into a single one with shape[100 / 512]
## Add/Normalize
## Feed Forward
## Output

# Mechanism
the way it works is every token get 3 embedding [key](#Key), [value](#Value), [query](#Query).

### Positional Embedding
The positional embedding of all tokens in the vocab is made during the boot up of the model.
This follows the same logic as the contextualize embedding, we simply make a matrix with the shape [dimension of the model / vocab size], and we take the scalar at the index of the token.
To get this matrix we use the [Sinusoidal Formula](#sinusoidal-formula) to get the positional embedding of all vectors

#### Sinusoidal Formula
the formula give us a relational value for each token so if we multiply the value of 2 tokens it give us the cosine of the difference between 2 tokens.
This follow the trigonometrical rule of cos(A)*cos(B) + sin(A)*sin(B) = cos(A - B), this entire thing is for the vectors to be spatially aware of where they are in relation to other vectors.
so we loop for i from 0 to dimension of model:
if i is even  we do PE(pos, 2i) = cos(pos / 10000*(2i / d-model))
if i is odd we do PE(pos, 2i + 1) = sin(pos / 10000*(2i / d-model))
where pos is the position of the token inside the  query, we do limit pos to context window, because the moment we surpass it the first token is forgotten.
