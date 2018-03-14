import os
import argparse
import numpy as np


# pylint: disable=invalid-name


UNK_TAG = '<unk>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        id2word = [word_count.rstrip().split(' ')[0] for word_count in f] + [UNK_TAG]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            word, *vec = line.rstrip().split(' ')
            vectors[word] = [float(val) for val in vec]

    vocab_size = len(id2word)
    word2id = {w: idx for idx, w in enumerate(id2word)}

    vector_dim = len(vectors[id2word[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        # if word == UNK_TAG:
        #     continue
        W[word2id[word], :] = v
        # try:
        #     W[word2id[word], :] = v
        # except KeyError:
        #     W[-1, :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, word2id)

def evaluate_vectors(W, word2id):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = './eval/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0     # count correct semantic questions
    correct_syn = 0     # count correct syntactic questions
    correct_tot = 0     # count correct questions
    count_sem = 0       # count all semantic questions
    count_syn = 0       # count all syntactic questions
    count_tot = 0       # count all questions
    full_count = 0      # count all questions, including those with unknown words

    for f_idx, fname in enumerate(filenames):
        with open(os.path.join(prefix, fname), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            known_data = [x_inst for x_inst in full_data
                          if all(word in word2id for word in x_inst)]
            # known_count += len(full_data)
            data = [[word if word in word2id else UNK_TAG for word in x_inst]
                    for x_inst in full_data]

        indices = np.array([[word2id[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for itr in range(num_iter):
            subset = np.arange(itr*split_size, min((itr + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                        +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for b_idx, batch in enumerate(subset):
                dist[ind1[batch], b_idx] = -np.Inf
                dist[ind2[batch], b_idx] = -np.Inf
                dist[ind3[batch], b_idx] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if f_idx < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("{}:".format(fname))
        print("ACCURACY TOP1: {:.2f}% ({:d}/{:d})\tfor {:d}/{:d} known words".format(
            np.mean(val) * 100, np.sum(val), len(val), len(known_data), len(full_data)))

    print("Questions seen/total: {:.2f}% ({:d}/{:d})".format(
        100 * count_tot / float(full_count), count_tot, full_count))
    print("Semantic accuracy: {:.2f}%  ({:d}/{:d})".format(
        100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print("Syntactic accuracy: {:.2f}%  ({:d}/{:d})".format(
        100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print("Total accuracy: {:.2f}%  ({:d}/{:d})".format(
        100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
