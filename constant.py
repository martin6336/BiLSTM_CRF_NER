# -*- coding:utf8 -*-
import os

config_file = "../config_files/config"
apply_file = os.path.join("../data", "example.dev")
train_file = os.path.join("../data", "example.train")
test_file = os.path.join("../data", "example.test")

config = {
    # config for the model
    "seg_dim":      20,                              # Embedding size for segmentation, 0 if not used
    "char_dim":     100,                             # Embedding size for characters
    "lstm_dim":     100,                             # Num of hidden units in LSTM
    "tag_schema":   "iob",                           # tagging schema iobes or iob

    # config for training
    "clean":        True,                            # clean train folder
    "clip":         5,                               # Gradient clip
    "batch_size":   20,                              # batch size
    "max_epoch":    1,                               # maximum training epochs
    "steps_check":  100,                             # steps per checkpoint
    "dropout":      0.5,                             # Dropout rate
    "lr":           0.001,                           # Initial learning rate
    "optimizer":    "adam",                          # Optimizer for training
    "pre_emb":      True,                            # Whether use pre-trained embedding
    "zeros":        False,                           # Whether replace digits with zero
    "lower":        True,                            # Whether lower case

    "map_file":     "../config_files/maps.pkl",                   # file for maps
    "emb_file":     "../config_files/wiki_100.utf8",              # Path for pre_trained embedding

    "result_path":  "../result",                     # Path for results
    "log_path":     "../log",                        # File for log

    "vocab_file":   "vocab.json",                    # File for vocab
    "summary_path": "summary",                       # Path to store summaries

    "ckpt_path":    "../ckpt",                       # Path to save model
}
