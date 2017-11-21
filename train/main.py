# -*- coding:utf8 -*-

import itertools
import os
import pickle

import numpy as np
import tensorflow as tf
from auxiliary_scripts.data_utils import load_word2vec, BatchManager
from auxiliary_scripts.loader import augment_with_pretrained, prepare_dataset
from auxiliary_scripts.loader import load_sentences, update_tag_scheme, char_mapping, tag_mapping
from auxiliary_scripts.utils import get_logger, make_path, clean, create_model, save_model, test_ner, save_config

import constant
from auxiliary_scripts.model import Model


# config for the model
def add_config(config, char_to_id, tag_to_id):
    config["num_chars"] = len(char_to_id)
    config["num_tags"] = len(tag_to_id)
    return config


def evaluate(sess, model, name, data, id_to_tag, logger, config):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, config["result_path"])
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train(config):
    # load data sets
    train_sentences = load_sentences(constant.train_file, config["lower"], config["zeros"])
    test_sentences = load_sentences(constant.test_file, config["lower"], config["zeros"])

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, config["tag_schema"])
    update_tag_scheme(test_sentences, config["tag_schema"])

    # create maps if not exist
    if not os.path.isfile(config["map_file"]):
        # create dictionary for word
        if config["pre_emb"]:
            dico_chars_train = char_mapping(train_sentences, config["lower"])[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                config["emb_file"],
                list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences]))
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, config["lower"])

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(config["map_file"], "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(config["map_file"], "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    config = add_config(config, char_to_id, tag_to_id)
    make_path(config)
    logger = get_logger(config["log_path"] + "/train.log")

    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, config["lower"])
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, config["lower"])

    train_manager = BatchManager(train_data, config["batch_size"])
    test_manager = BatchManager(test_data, 100)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, config["ckpt_path"], load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(config["max_epoch"]):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % config["steps_check"] == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(iteration,
                                      step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "test", test_manager, id_to_tag, logger, config)
            if best:
                save_model(sess, model, config["ckpt_path"], logger)


if __name__ == "__main__":
    config = constant.config
    if config["clean"]:
        clean(config)
    train(config)
    save_config(config, constant.config_file)
