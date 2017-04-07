#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
GloVe wrapper
=============

Synopsis
--------
    examples:
    `````````
        python scripts/run_glove.py -te \
            --corpus-fpath ../word2vec_data/data_no_unk_tag.txt

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
import struct

from math import floor, log2

import numpy as np
import pandas as pd
import psutil
from IPython import embed  # , start_ipython
from IPython.display import display


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
    def __init__(self, argp, job_idx=None):
        self.config = {}
        self.embeds_fbasepath = ''
        self.vocab_fpath = ''
        self.cooccur_fpath = ''
        self.shuf_cooc_fpath = ''
        self.eval_fpath = ''

        self.corpus_fpath = ''

        self.memory = None
        self.num_threads = None
        self.cooccurrences = []
        self.id2word = []
        self.cooccurr_dic = {}
        self.cooccurrences_df = None

        self.argp = argp

        self._load_config()
        self._build_param_tag_paths(job_idx)

        if argp.corpus_fpath:
            self.corpus_fpath = argp.corpus_fpath
        else:
            self.corpus_fpath = os.path.join(DATA_PATH, self.config['corpus_fname'])

        self.set_ressources()   # TODO: handle manual change in argp and simultaneous process

    def _load_config(self):
        conf_fpath = self._get_param_tag_fpath(SCRIPT_PATH, CONF_FNAME, [self.argp.corpus])
        print("Config file used:", conf_fpath)
        with open(conf_fpath) as f:
            self.config = json.load(f)

    @staticmethod
    def _get_param_tag_fpath(path, fname, params):
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

    def _build_param_tag_paths(self, job_idx=None):

        idx_name = 'num'

        vocab_params = [(self.argp.corpus,),
                        ('cnt', self.config['voc_min_cnt'])]
        data_params = vocab_params + [('win', self.config['win_size'])]
        model_params = data_params + [('dim', self.config['embeds_dim']),
                                      ('itr', self.config['max_iter']),
                                      ('xmx', self.config['x_max'])]
        if job_idx is not None:
            model_params += [(idx_name, job_idx)]

        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(EVAL_PATH, exist_ok=True)

        self.vocab_fpath = self._get_param_tag_fpath(DATA_PATH, VOCAB_FNAME, vocab_params)
        self.cooccur_fpath = self._get_param_tag_fpath(DATA_PATH, COOCCURR_FNAME, data_params)
        self.shuf_cooc_fpath = self._get_param_tag_fpath(DATA_PATH, SHUF_COOC_FNAME, data_params)
        self.embeds_fbasepath = self._get_param_tag_fpath(MODEL_PATH, EMBEDS_FNAME, model_params)
        # self.eval_fpath = self._get_param_tag_fpath(EVAL_PATH, EVAL_FNAME, model_params)

        # TODO: make change for all files. Make next task taking in_fname into account
        #       e.g.: shuf_cooc*_num1.bin >> num1 is detected by train()
        self.embeds_fbasepath = increment_idx_existing_fname(self.embeds_fbasepath+'.txt',
                                                             idx_name)[:-len('.txt')]
        # self.eval_fpath = increment_idx_existing_fname(self.eval_fpath, idx_name)
        self.eval_fpath = os.path.join(     # TODO: remove this temp fix with global solution
            EVAL_PATH,
            'eval' + os.path.split(self.embeds_fbasepath)[1][len(EMBEDS_FNAME):] + '.txt')

        # self.model_fpaths = [self.embeds_bin_fpath, self.embeds_txt_fpath]
        # self.all_fpaths = self.model_fpaths + \
        #     [self.vocab_fpath, self.cooccur_fpath, self.shuf_cooc_fpath, self.eval_fpath]

    def set_ressources(self, num_threads=None, memory=None, num_jobs=1):
        if memory is None:
            memory = psutil.virtual_memory().available / 1024**3
        if num_threads is None:
            num_threads = os.cpu_count()
        self.memory = 2**floor(log2(memory / num_jobs))
        self.num_threads = num_threads / num_jobs

    def _fix_vocab(self):
        '''Remove missing word (space char) in the vocab file.'''
        with open(self.vocab_fpath) as f:
            vocab = [l.rstrip('\n').split(' ') for l in f]
        vocab_cln = [(word, cnt) for (word, cnt) in vocab if word != '']
        if len(vocab_cln) != len(vocab):
            print("{} word(s) removed from the vocab!".format(len(vocab) - len(vocab_cln)))
            with open(self.vocab_fpath, 'w') as f:
                f.writelines('{} {}\n'.format(word, cnt) for (word, cnt) in vocab_cln)

    @staticmethod
    def _run_command(command, name=None, stdin=None, stdout=None):
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
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)
        # Remove missing word (space char) in the vocab file.
        self._fix_vocab()

    def build_cooccurr(self):
        command = [COOCCUR,
                   '-memory', self.memory,
                   '-vocab-file', self.vocab_fpath,
                   '-verbose', self.config['verbose'],
                   '-window-size', self.config['win_size']]
        with open(self.corpus_fpath) as f_in, open(self.cooccur_fpath, 'w') as f_out:
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)

    def build_shuf_cooc(self):
        command = [SHUFFLE,
                   '-memory', self.memory,
                   '-verbose', self.config['verbose'],
                   '-window-size', self.config['win_size']]
        with open(self.cooccur_fpath) as f_in, open(self.shuf_cooc_fpath, 'w') as f_out:
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)

    def train(self):
        command = [GLOVE,
                   '-save-file', self.embeds_fbasepath,
                   '-threads', self.num_threads,
                   '-input-file', self.shuf_cooc_fpath,
                   '-x-max', self.config['x_max'],
                   '-iter', self.config['max_iter'],
                   '-vector-size', self.config['embeds_dim'],
                   '-binary', self.config['binary'],
                   '-vocab-file', self.vocab_fpath,
                   '-verbose', self.config['verbose']]
        self._run_command(lst2str_lst(command))

    def sim_eval(self):
        command = ['python', EVAL,
                   '--vocab_file', self.vocab_fpath,
                   '--vectors_file', self.embeds_fbasepath+'.txt']
        params = join_list(sorted(self.config.items()))
        # Print parameters as header to evaluation output file
        with open(self.eval_fpath, 'w') as f_out:
            f_out.write(params + '\n')
            f_out.write('==========\n')
        with open(self.eval_fpath, 'a') as f_out:
            self._run_command(lst2str_lst(command), name=EVAL, stdout=f_out)

    def pre_process(self):
        self.build_vocab()
        self.build_cooccurr()
        self.build_shuf_cooc()

    def full_train(self):
        self.pre_process()
        self.train()

    def read_cooccurrences(self):
        struct_fmt = 'iid'  # int, int, double
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        def read_chunks(f, length):
            while True:
                data = f.read(length)
                if not data:
                    break
                yield data
        with open(os.path.join(self.cooccur_fpath), 'rb') as f:
            self.cooccurrences = [struct_unpack(chunk) for chunk in read_chunks(f, struct_len)]

    def _set_id2word(self):
        with open(os.path.join(self.vocab_fpath)) as f:
            self.id2word = [l.rstrip('\n').split(' ')[0] for l in f]

    def _set_cooccurr_dic(self):
        self.cooccurr_dic = {tuple([self.id2word[w_id - 1] for w_id in word_ids]): cnt
                             for (*word_ids, cnt) in self.cooccurrences}

    def get_cooccurrence_btwn(self, wrd1, wrd2):
        return self.cooccurr_dic.get((wrd1, wrd2))

    def get_cooccurrences(self, word):
        return [{word_pair: self.cooccurr_dic[word_pair]}
                for word_pair in self.cooccurr_dic
                if word == word_pair[0]]

    def _set_cooccurrences_df(self):
        '''Display cooccurrence matrix using Pandas.'''
        cooccurrences_df = pd.DataFrame(self.cooccurrences, columns=['w1', 'w2', 'count'])
        cooccurrences_df = cooccurrences_df.pivot_table(index='w1', columns='w2')['count']
        del cooccurrences_df.index.name, cooccurrences_df.columns.name
        cooccurrences_df = cooccurrences_df.fillna(0)
                                        # Remove inferior triangle (sym mtrx)
        cooccurrences_df = pd.DataFrame(np.triu(cooccurrences_df.as_matrix()))
        cooccurrences_df.columns, cooccurrences_df.index = self.id2word, self.id2word
        cooccurrences_df[cooccurrences_df == 0] = ''
        self.cooccurrences_df = cooccurrences_df

    def setup_cooccurr_analysis(self):
        self.read_cooccurrences()
        self._set_id2word()
        self._set_cooccurr_dic()
        self._set_cooccurrences_df()

    def display_cooccurrence(self):
        with pd.option_context('max_rows', 20,
                               'display.float_format', '{:,.1f}'.format,
                               'display.max_colwidth', 3):
            display(self.cooccurrences_df)


def lst2str_lst(lst):
    return [str(el) for el in lst]


def join_list(lst, sep=' '):
    return sep.join(lst2str_lst(lst))


def increment_idx_existing_fname(fpath, idx_name):  # TODO: mv fct in ext module (to incl in repo)
    if os.path.isfile(fpath):
        path, fname = os.path.split(fpath)
        (basename, ext) = os.path.splitext(fname)
        bname_splits = basename.split('_')
        if idx_name in bname_splits[-1]:
            idx = int(bname_splits[-1][len(idx_name):])
            new_fname = '_'.join(bname_splits[:-1] + [idx_name + str(idx + 1)])
        else:
            new_fname = '_'.join(bname_splits + [idx_name + '1'])
        new_fpath = os.path.join(path, new_fname + ext)
        return increment_idx_existing_fname(new_fpath, idx_name)
    return fpath


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--corpus', choices=['big', 'toy'], default='big',
                        help='Training dataset name.')

    parser.add_argument('--corpus-fpath',
                        help='Training dataset filepath.')

    parser.add_argument('-n', '--num-jobs', type=int, default=1,
                        help='Perform multiple training pass in parallel.')

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
    group.add_argument('-a', '--analysis', action='store_true',
                       help='Start analysis interactive mode.')
    # group.add_argument('-e', '--eval', dest='model_fname',
    #                    help='Perform the evaluation test based on the model given as argument.')

    argp = parser.parse_args(args)

    # if argp.embeds_src == 'w2v_c' and argp.w2vc_name == '' and argp.train:
    #     parser.error("--embeds-src='w2v_c' requires -w W2VC_NAME.")

    return argp


def full_process(argp, job_idx=None):

    glove = GloVe(argp, job_idx)

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
        glove.sim_eval()

    if argp.analysis:
        glove.setup_cooccurr_analysis()
        embed()


def main(argp):

    # TODO: allow to use different parameters per job
    if argp.num_jobs > 1:
        for job_idx in range(argp.num_jobs):
            full_process(argp, job_idx=job_idx+1)
    else:
        full_process(argp)


if __name__ == '__main__':
    main(get_args())
