# Evaluator

Calculates word error rates and handles conversion of CTC predictions to numeric tokens.

### Evaluator.sequenceErrorRate(target, prediction)

Calculates the word error rates based on the target and the predicted inputs.

`target` and `prediction` are a table of words such as ```{"example", "of", "words"}```.