#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
GloVe wrapper
=============

Synopsis
--------
    examples:
    `````````
        ./scripts/run_glove.py -texj 3 --corpus-fpath ../word2vec_data/data_no_unk_tag.txt
        ./scripts/run_glove.py -feac toy --corpus-fpath ./data_toy/data_toy.txt

Authors
-------
* Marc Evrard         (<marc.evrard@gmail.com>)

License
-------
Copyright 2017 Marc Evrard

Licensed under the Apache License, Version 2.0 (the "License")
http://www.apache.org/licenses/LICENSE-2.0
'''

import argparse
import json
import os
import struct
import subprocess
import sys
from math import floor, log2

import numpy as np
import pandas as pd
import psutil
from IPython import embed  # , start_ipython
from IPython.display import display

import embedding_tools as emb


PATHS_FNAME = 'paths.json'
CONF_FNAME = 'config.json'

# TODO: Create parent classes for wvec and derive this module from them!

class Option:
    def __init__(self, argp, job_idx=None):
        self.config = {}
        self.embeds_fbasepath = ''
        self.vocab_fpath = ''
        self.cooccur_fpath = ''
        self.shuf_cooc_fpath = ''
        self.eval_fpath = ''
        self.memory = None
        self.num_threads = None

        self.argp = argp

        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.paths = paths = self._load_paths()

        (self.data_path, self.model_path, self.bin_path, self.eval_path, self.vocab_fname,
         self.cooccurr_fname, self.shuf_cooc_fname, self.embeds_fname, self.eval_fname,
         self.eval
        ) = (None,) * 10

        for (key, value) in paths.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.vocab_count = os.path.join(self.bin_path, 'vocab_count')
        self.cooccur = os.path.join(self.bin_path, 'cooccur')
        self.shuffle = os.path.join(self.bin_path, 'shuffle')
        self.glove = os.path.join(self.bin_path, 'glove')

        self.config = config = self._load_config()

        (self.corpus_fname, self.verbose, self.min_count, self.embeds_dim, self.max_iter,
         self.win_size, self.binary, self.x_max
        ) = (None,) * 8

        for (key, value) in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self._build_param_tag_paths(job_idx)

        if argp.corpus_fpath:
            self.corpus_fpath = argp.corpus_fpath
        else:
            self.corpus_fpath = os.path.join(paths['data_path'], self.corpus_fname)

        self.set_ressources(argp.num_threads)   # TODO: handle simultaneous process(?)

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.eval_path, exist_ok=True)

    def _load_paths(self):
        with open(os.path.join(self.script_path, PATHS_FNAME)) as f:
            return json.load(f)

    def _load_config(self):
        conf_fpath = self._get_param_tag_fpath(self.script_path, CONF_FNAME,
                                               [self.argp.corpus_type])
        print("Config file used:", conf_fpath)
        with open(conf_fpath) as f:
            return json.load(f)

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

        vocab_params = [(self.argp.corpus_type,),
                        ('cnt', self.min_count)]
        data_params = vocab_params + [('win', self.win_size)]
        model_params = data_params + [('dim', self.embeds_dim),
                                      ('itr', self.max_iter),
                                      ('xmx', self.x_max)]
        if job_idx is not None:
            model_params += [(idx_name, job_idx)]

        self.vocab_fpath = self._get_param_tag_fpath(self.data_path, self.vocab_fname,
                                                     vocab_params)
        self.cooccur_fpath = self._get_param_tag_fpath(self.data_path, self.cooccurr_fname,
                                                       data_params)
        self.shuf_cooc_fpath = self._get_param_tag_fpath(self.data_path, self.shuf_cooc_fname,
                                                         data_params)
        self.embeds_fbasepath = self._get_param_tag_fpath(self.model_path, self.embeds_fname,
                                                          model_params)
        # self.eval_fpath = self._get_param_tag_fpath(self.eval_path, EVAL_FNAME, model_params)

        # TODO: make change for all files. Make next task taking in_fname into account
        #       e.g.: shuf_cooc*_num1.bin >> num1 is detected by train()
        if self.argp.train or self.argp.full_train:    # only at training time (allow eval only)
            self.embeds_fbasepath = increment_idx_existing_fname(self.embeds_fbasepath+'.txt',
                                                                 idx_name)[:-len('.txt')]
        # self.eval_fpath = increment_idx_existing_fname(self.eval_fpath, idx_name)
        self.eval_fpath = os.path.join(     # TODO: remove this temp fix with global solution
            self.eval_path,
            'eval' + os.path.split(self.embeds_fbasepath)[1][len(self.embeds_fname):] + '.txt')

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


class GloVe:
    def __init__(self, opts):
        self.opts = opts

    def _fix_vocab(self):
        '''Remove missing word (space char) in the vocab file.'''
        with open(self.opts.vocab_fpath) as f:
            vocab = [l.rstrip('\n').split(' ') for l in f]
        vocab_cln = [(word, cnt) for (word, cnt) in vocab if word != '']
        if len(vocab_cln) != len(vocab):
            print("{} word(s) removed from the vocab!".format(len(vocab) - len(vocab_cln)))
            with open(self.opts.vocab_fpath, 'w') as f:
                f.writelines('{} {}\n'.format(word, cnt) for (word, cnt) in vocab_cln)

    @staticmethod
    def _run_command(command, name=None, stdin=None, stdout=None):
        print(join_list(command))
        try:
            subprocess.run(command, stdin=stdin, stdout=stdout, check=True)
        except FileNotFoundError:
            sys.exit("Error! Is GloVe installed? If not: run `make`")
        if name is None:
            name = command[0]
        print("'{}' done.\n".format(name))

    def build_vocab(self):
        opts = self.opts
        command = [opts.vocab_count,
                   '-min-count', opts.min_count,
                   '-verbose', opts.verbose]
        with open(opts.corpus_fpath, 'r') as f_in, open(opts.vocab_fpath, 'w') as f_out:
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)
        # Remove missing word (space char) in the vocab file.
        self._fix_vocab()

    def build_cooccurr(self):
        opts = self.opts
        command = [self.opts.cooccur,
                   '-memory', opts.memory,
                   '-vocab-file', opts.vocab_fpath,
                   '-verbose', opts.verbose,
                   '-window-size', opts.win_size]
        with open(opts.corpus_fpath) as f_in, open(opts.cooccur_fpath, 'w') as f_out:
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)

    def build_shuf_cooc(self):
        opts = self.opts
        command = [opts.shuffle,
                   '-memory', opts.memory,
                   '-verbose', opts.verbose,
                   '-window-size', opts.win_size]
        with open(opts.cooccur_fpath) as f_in, open(opts.shuf_cooc_fpath, 'w') as f_out:
            self._run_command(lst2str_lst(command), stdin=f_in, stdout=f_out)
        os.remove(opts.cooccur_fpath)

    def train(self):
        opts = self.opts
        command = [opts.glove,
                   '-save-file', opts.embeds_fbasepath,
                   '-threads', opts.num_threads,
                   '-input-file', opts.shuf_cooc_fpath,
                   '-x-max', opts.x_max,
                   '-iter', opts.max_iter,
                   '-vector-size', opts.embeds_dim,
                   '-binary', opts.binary,
                   '-vocab-file', opts.vocab_fpath,
                   '-verbose', opts.verbose]
        self._run_command(lst2str_lst(command))

    def sim_eval(self):
        # TODO: apply same tests in GloVe and W2V
        opts = self.opts
        command = ['python3', opts.eval,
                   '--vocab_file', opts.vocab_fpath,
                   '--vectors_file', opts.embeds_fbasepath+'.txt']
        params = join_list(sorted(opts.config.items()))
        # Print parameters as header to evaluation output file
        with open(opts.eval_fpath, 'w') as f_out:
            f_out.write(params + '\n')
            f_out.write('==========\n')
        with open(opts.eval_fpath, 'a') as f_out:
            self._run_command(lst2str_lst(command), name=opts.eval, stdout=f_out)

    def pre_process(self):
        self.build_vocab()
        self.build_cooccurr()
        self.build_shuf_cooc()

    def full_train(self):
        self.pre_process()
        self.train()


class Analysis:
    def __init__(self, opts):
        self.opts = opts

        self.cooccurr_dic = {}
        self.shuf_cooccurr = []
        self.cooccurrences_df = None
        self.id2word_cooc = []

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
        with open(self.opts.shuf_cooc_fpath, 'rb') as f:
            self.shuf_cooccurr = [struct_unpack(chunk) for chunk in read_chunks(f, struct_len)]

    def _set_id2word_cooc(self):
        with open(os.path.join(self.opts.vocab_fpath)) as f:
            self.id2word_cooc = [l.rstrip('\n').split(' ')[0] for l in f]

    def _set_cooccurr_dic(self):
        self.cooccurr_dic = {tuple([self.id2word_cooc[w_id - 1] for w_id in word_ids]): cnt
                             for (*word_ids, cnt) in self.shuf_cooccurr}

    def get_cooccurrence_btwn(self, wrd1, wrd2):
        return self.cooccurr_dic.get((wrd1, wrd2))

    def get_cooccurrences(self, word):
        return [{word_pair: self.cooccurr_dic[word_pair]}
                for word_pair in self.cooccurr_dic
                if word == word_pair[0]]

    def _set_cooccurrences_df(self):
        '''Display cooccurrence matrix using Pandas.'''
        cooccurrences_df = pd.DataFrame(self.shuf_cooccurr, columns=['w1', 'w2', 'count'])
        cooccurrences_df = cooccurrences_df.pivot_table(index='w1', columns='w2')['count']
        del cooccurrences_df.index.name, cooccurrences_df.columns.name
        cooccurrences_df = cooccurrences_df.fillna(0)
                                        # Remove inferior triangle (sym mtrx)
        cooccurrences_df = pd.DataFrame(np.triu(cooccurrences_df.as_matrix()))
        cooccurrences_df.columns, cooccurrences_df.index = self.id2word_cooc, self.id2word_cooc
        cooccurrences_df[cooccurrences_df == 0] = ''
        self.cooccurrences_df = cooccurrences_df

    def setup_cooccurr_analysis(self):
        self.read_cooccurrences()
        self._set_id2word_cooc()
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
    parser.add_argument('-c', '--corpus-type', choices=['big', 'toy'], default='big',
                        help='Training dataset name.')
    parser.add_argument('--corpus-fpath',
                        help='Training dataset filepath.')
    parser.add_argument('-j', '--num-jobs', type=int, default=1,
                        help='Set the number of successive jobs.')
    parser.add_argument('-i', '--data-info', default='',
                        help='Extra info used to describe and sort the current model.')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Perform the evaluation test.')
    parser.add_argument('-x', '--export-embeds', action='store_true',
                        help='Export embeddings and vocabulary to file.')
    parser.add_argument('-l', '--num-threads', type=int,
                        help='Limit the number of CPU threads.')
    parser.add_argument('-a', '--analysis', action='store_true',
                        help='Start analysis interactive mode.')

    group = parser.add_mutually_exclusive_group()   # required=True
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


def full_process(argp, job_idx=None):

    options = Option(argp, job_idx)

    glove = GloVe(options)
    analysis = Analysis(options)

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
        analysis.setup_cooccurr_analysis()
        embed()

    if argp.export_embeds:
        emb.conv_embeds(options.embeds_fbasepath + '.txt')
        # emb.rm_txt_bin_embeds(options.embeds_fbasepath)


def main(argp):

    # TODO: allow to use different parameters per job
    if argp.num_jobs > 1:
        for job_idx in range(argp.num_jobs):
            full_process(argp, job_idx=job_idx+1)
    else:
        full_process(argp)


if __name__ == '__main__':
    try:
        main(get_args())
    except KeyboardInterrupt:
        sys.exit("\nProgram interrupted by user.\n")
