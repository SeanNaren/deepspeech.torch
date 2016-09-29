# Mapper

Defines how numeric indices are mapped to tokens and vice versa.

### Mapper:__init(dictPath)

Creates mappings based on the given dictionary file. The AN4 dictionary file can be seen [here](https://github.com/SeanNaren/deepspeech.torch/blob/master/dictionary).

### Mapper:encodeString(string)

Converts string into a set of tokens to be used as a label in training.

`string` string to be converted.

### Mapper:decodeOutput(predictions)

Converts predictions of the neural network into a sequence of tokens (characters) via a mapper.

`predictions` is a tensor of sequence likelihood vectors given by the neural network.

### Mapper:tokensToText(tokens)

Using the mapper converts the tokens into readable text.

`tokens` A set of numeric tokens to convert into readable text.
