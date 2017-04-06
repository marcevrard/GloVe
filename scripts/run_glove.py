#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
GloVe wrapper
=============

Synopsis
--------
    examples:
    `````````
        python scripts/run_glove.py -e -c ../word2vec_data/data_no_unk_tag.txt

Authors
-------
* Marc Evrard         (<marc.evrard@gmail.com>)

License
-------
Licensed under the Apache License, Version 2.0 (the "License"):

http://www.apache.org/licenses/LICENSE-2.0
'''

import os
import subprocess
import argparse
import json

from math import floor, log2

import psutil


DATA_PATH = './data'
MODEL_PATH = './model'
BIN_PATH = './build'
EVAL_PATH = './eval/results'
SCRIPT_PATH = './scripts'

CONF_FNAME = 'config.json'
VOCAB_FNAME = 'vocab.txt'
COOCCURR_FNAME = 'cooccurrence.bin'
SHUF_COOC_FNAME = 'shuf_cooccurr.bin'
EMBEDS_FNAME = 'vectors_glove'
EVAL_FNAME = 'eval.txt'

VOCAB_COUNT = os.path.join(BIN_PATH, 'vocab_count')
COOCCUR = os.path.join(BIN_PATH, 'cooccur')
SHUFFLE = os.path.join(BIN_PATH, 'shuffle')
GLOVE = os.path.join(BIN_PATH, 'glove')
EVAL = './eval/python/evaluate.py'


class GloVe:
    def __init__(self, argp, num=None):
        self.config = {}
        self.embeds_fpath = ''
        self.vocab_fpath = ''
        self.cooccur_fpath = ''
        self.shuf_cooc_fpath = ''
        self.eval_fpath = ''
        self.corpus_fpath = ''

        self.memory = None
        self.num_threads = None

        self.argp = argp

        self.load_config()
        self.build_paths(num)

        self.set_ressources()   # TODO: handle manual change in argp and simultaneous process

    def load_config(self):
        conf_fpath = self.get_param_tagged_fpath(SCRIPT_PATH, CONF_FNAME, [self.argp.corpus])
        print("Config file used:", conf_fpath)
        with open(conf_fpath) as f:
            self.config = json.load(f)

    @staticmethod
    def get_param_tagged_fpath(path, fname, params):
        assert not isinstance(params, str)
        suffix = '_'.join([join_list(param_pair, sep='') for param_pair in params])
        (basename, ext) = os.path.splitext(fname)
        new_basename = '{}_{}{}'.format(basename, suffix, ext)
        return os.path.join(path, new_basename)

    # def clean_old(self, fpaths):
    #     assert not isinstance(fpaths, str)
    #     for fpath in fpaths:
    #         subprocess.run('rm', fpath)
    #     print("Cleaning of {} done.".format(fpaths))

    def build_paths(self, num=None):

        vocab_params = [(self.argp.corpus,),
                        ('cnt', self.config['voc_min_cnt'])]
        data_params = vocab_params + [('win', self.config['win_size'])]
        model_params = data_params + [('dim', self.config['embeds_dim']),
                                      ('itr', self.config['max_iter']),
                                      ('xmx', self.config['x_max'])]
        if num is not None:
            model_params += [('num', num)]

        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(EVAL_PATH, exist_ok=True)

        self.embeds_fpath = self.get_param_tagged_fpath(MODEL_PATH, EMBEDS_FNAME, model_params)
        self.vocab_fpath = self.get_param_tagged_fpath(DATA_PATH, VOCAB_FNAME, vocab_params)
        self.cooccur_fpath = self.get_param_tagged_fpath(DATA_PATH, COOCCURR_FNAME, data_params)
        self.shuf_cooc_fpath = self.get_param_tagged_fpath(DATA_PATH, SHUF_COOC_FNAME, data_params)
        self.eval_fpath = self.get_param_tagged_fpath(EVAL_PATH, EVAL_FNAME, model_params)

        self.corpus_fpath = os.path.join(DATA_PATH, self.config['corpus_fname'])

        # self.model_fpaths = [self.embeds_bin_fpath, self.embeds_txt_fpath]
        # self.all_fpaths = self.model_fpaths + \
        #     [self.vocab_fpath, self.cooccur_fpath, self.shuf_cooc_fpath, self.eval_fpath]

    def set_ressources(self):
        available_memory = psutil.virtual_memory().available / 1024**3
        self.memory = 2**floor(log2(available_memory))
        self.num_threads = os.cpu_count()

    def fix_vocab(self):
        '''Remove missing word (space char) in the vocab file.'''
        with open(self.vocab_fpath) as f:
            vocab = [l.rstrip('\n').split(' ') for l in f]
        vocab_cln = [(word, cnt) for (word, cnt) in vocab if word != '']
        if len(vocab_cln) != len(vocab):
            print("{} word(s) removed from the vocab!".format(len(vocab) - len(vocab_cln)))
            with open(self.vocab_fpath) as f:
                f.writelines('{} {}\n'.format(word, cnt) for (word, cnt) in vocab_cln)

    @staticmethod
    def run_command(command, name=None, stdin=None, stdout=None):
        print(join_list(command))
        subprocess.run(command, stdin=stdin, stdout=stdout)
        if name is None:
            name = command[0]
        print("'{}' done.\n".format(name))

    def build_vocab(self):
        command = [VOCAB_COUNT,
                   '-min-count', self.config['voc_min_cnt'],
                   '-verbose', self.config['verbose']]
        with open(self.corpus_fpath, 'r') as f_in, open(self.vocab_fpath, 'w') as f_out:
            self.run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)
        # Remove missing word (space char) in the vocab file.
        self.fix_vocab()

    def build_cooccurr(self):
        command = [COOCCUR,
                   '-memory', self.memory,
                   '-vocab-file', self.vocab_fpath,
                   '-verbose', self.config['verbose'],
                   '-window-size', self.config['win_size']]
        with open(self.corpus_fpath) as f_in, open(self.cooccur_fpath, 'w') as f_out:
            self.run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)

    def build_shuf_cooc(self):
        command = [SHUFFLE,
                   '-memory', self.memory,
                   '-verbose', self.config['verbose'],
                   '-window-size', self.config['win_size']]
        with open(self.cooccur_fpath) as f_in, open(self.shuf_cooc_fpath, 'w') as f_out:
            self.run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)

    def train(self):
        command = [GLOVE,
                   '-save-file', self.embeds_fpath,
                   '-threads', self.num_threads,
                   '-input-file', self.shuf_cooc_fpath,
                   '-x-max', self.config['x_max'],
                   '-iter', self.config['max_iter'],
                   '-vector-size', self.config['embeds_dim'],
                   '-binary', self.config['binary'],
                   '-vocab-file', self.vocab_fpath,
                   '-verbose', self.config['verbose']]
        self.run_command(lst2str_lst(command))

    def eval(self):
        command = ['python', EVAL,
                   '--vocab_file', self.vocab_fpath,
                   '--vectors_file', self.embeds_fpath+'.txt']
        params = join_list(sorted(self.config.items()))
        # Print parameters as header to evaluation output file
        with open(self.eval_fpath, 'w') as f_out:
            f_out.write(params + '\n')
            f_out.write('==========\n')
        with open(self.eval_fpath, 'a') as f_out:
            self.run_command(lst2str_lst(command), name=EVAL, stdout=f_out)

    def pre_process(self):
        self.build_vocab()
        self.build_cooccurr()
        self.build_shuf_cooc()

    def full_train(self):
        self.pre_process()
        self.train()


def lst2str_lst(lst):
    return [str(el) for el in lst]


def join_list(lst, sep=' '):
    return sep.join(lst2str_lst(lst))


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--corpus', choices=['big', 'toy'], default='big',
                        help='Training dataset.')

    parser.add_argument('-i', '--data-info', default='',
                        help='Extra info used to describe and sort the current model.')

    parser.add_argument('-e', '--eval', action='store_true',
                        help='Perform the evaluation test.')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-p', '--pre-process', action='store_true',
                       help='Train the models.')
    group.add_argument('-t', '--train', action='store_true',
                       help='Train the models.')
    group.add_argument('-f', '--full-train', action='store_true',
                       help='Pre-process and train the models.')
    # group.add_argument('-e', '--eval', dest='model_fname',
    #                    help='Perform the evaluation test based on the model given as argument.')

    argp = parser.parse_args(args)

    # if argp.embeds_src == 'w2v_c' and argp.w2vc_name == '' and argp.train:
    #     parser.error("--embeds-src='w2v_c' requires -w W2VC_NAME.")

    return argp


def main(argp):

    glove = GloVe(argp, num=1)

    if argp.pre_process:
        print("\n** PRE-PROCESSING **\n")
        glove.pre_process()
    elif argp.train:
        print("\n** TRAIN MODEL **\n")
        glove.train()
    elif argp.full_train:
        print("\n** PRE-PROCESS & TRAIN MODEL **\n")
        glove.full_train()

    if argp.eval:
        print("\n** EVALUATE MODEL **\n")
        glove.eval()


if __name__ == '__main__':
    main(get_args())
