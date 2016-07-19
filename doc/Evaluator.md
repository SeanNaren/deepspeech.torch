# Evaluator

Calculates word error rates and handles conversion of CTC predictions to numeric tokens.

### Evaluator.sequenceErrorRate(target, prediction)

Calculates the word error rates based on the target and the predicted inputs.

`target` and `prediction` are a table of words such as ```{"example", "of", "words"}```.

### Evaluator.predict2tokens(predictions, mapper)

Converts predictions of the neural network into a sequence of tokens (characters) via a mapper.

`predictions` is a tensor of sequence likelihood vectors given by the neural network.

`mapper` Defines how numeric values are converted to readable tokens (see [Mapper](https://github.com/SeanNaren/CTCSpeechRecognition/doc/Mapper.md)).
