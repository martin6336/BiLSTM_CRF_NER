# -*- coding:utf-8 -*-

import pickle

import tensorflow as tf
from auxiliary_scripts.data_utils import load_word2vec, BatchManager
from auxiliary_scripts.loader import load_sentences, prepare_dataset
from auxiliary_scripts.utils import get_logger, create_model, load_config

import constant
from auxiliary_scripts.model import Model

if __name__ == "__main__":
    config = load_config(constant.config_file)
    logger = get_logger(config["log_path"] + "/apply.log")

    with open(config["map_file"], "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    apply_sentences = load_sentences(constant.apply_file, config["lower"], config["zeros"])
    apply_data = prepare_dataset(apply_sentences, char_to_id, tag_to_id, config["lower"])
    apply_manager = BatchManager(apply_data, 100)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, config["ckpt_path"], load_word2vec, config, id_to_char, logger)
        for batch in apply_manager.iter_batch(shuffle=False):
            lengths, scores = model.run_step(sess, False, batch)
            trans = model.trans.eval()
            batch_paths = model.decode(scores, lengths, trans)
            for i in range(100):
                for item in batch[0][i]:
                    print item,
                print " "
                tags = [id_to_tag[idx] for idx in batch_paths[i]]
                print tags
            break
