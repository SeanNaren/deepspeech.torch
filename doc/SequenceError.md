# SequenceError

Calculates word error rates and handles conversion of CTC predictions to numeric tokens.

### SequenceError.sequenceErrorRate(target, prediction)

Calculates the error rates based on the target and the predicted inputs.

`target` and `prediction` are inputs of strings or tables.

### SequenceError:calculateCER(targetTranscript, predictTranscript)

`targetTranscript` and `predictTranscript` are two strings, returns the Character Error Rate.

### SequenceError:calculateWER(targetTranscript, predictTranscript)

`targetTranscript` and `predictTranscript` are two strings, returns the Word Error Rate.