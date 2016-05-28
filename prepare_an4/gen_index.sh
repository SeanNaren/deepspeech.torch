#!/bin/sh
TRANS=etc/an4_train.transcription
FIELD=etc/an4_train.fileids
INDEX=train_index.txt
awk 'NR==FNR{a[NR]=$0;next}{printf "%s.wav@%s@\n", a[FNR], $0}' $1/$FIELD $1/$TRANS > $INDEX

TRANS=etc/an4_test.transcription
FIELD=etc/an4_test.fileids
INDEX=test_index.txt
awk 'NR==FNR{a[NR]=$0;next}{printf "%s.wav@%s@\n", a[FNR], $0}' $1/$FIELD $1/$TRANS > $INDEX
