#!/usr/bin/env bash

  python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=True \
    --do_train=False   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=CVA-data   \
    --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=64   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=5.0   \
    --output_dir=./output/result_dir


perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt
