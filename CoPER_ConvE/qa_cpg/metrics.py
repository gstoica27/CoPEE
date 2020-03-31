from __future__ import absolute_import, division, print_function

import logging
import os
import numpy as np
import tensorflow as tf
import argparse
import sys
from collections import Counter

__all__ = ['ranking_and_hits']

logger = logging.getLogger(__name__)


def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")


def ranking_and_hits(model, results_dir, data_iterator_handle, name, session=None, hits_to_compute=(1, 3, 5, 10, 20),
                     enable_write_to_file=False):
    os.makedirs(results_dir, exist_ok=True)
    logger.info('')
    logger.info('-' * 50)
    logger.info(name)
    logger.info('-' * 50)
    logger.info('')

    hits = {hits_level: [] for hits_level in hits_to_compute}

    ranks = []

    stopped = False
    count = 0
    while not stopped:
        try:
            e1, e2, rel, rel_multi, pred = session.run(
                (model.e1, model.e2, model.rel, model.rel_multi, model.predictions_all),
                feed_dict={model.input_iterator_handle: data_iterator_handle})

            target_values = pred[np.arange(0, len(pred)), rel]
            pred[rel_multi == 1] = -np.inf
            pred[np.arange(0, len(pred)), rel] = target_values
            count += e1.shape[0]
            for i in range(len(e1)):
                pred1_args = np.argsort(-pred[i])
                rank = int(np.where(pred1_args == rel[i])[0]) + 1
                ranks.append(rank)

                for hits_level in hits_to_compute:
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        except tf.errors.OutOfRangeError:
            stopped = True

    logger.info('Evaluated %d samples.' % count)

    # Save results.
    for hits_level in hits_to_compute:
        hits_value = np.mean(hits[hits_level])
        logger.info('Hits @%d: %10.6f', hits_level, hits_value)
        hits[hits_level] = hits_value
        # Write hits to respective files.
        if enable_write_to_file:
            hits_at_path = os.path.join(results_dir, 'hits_at_{}.txt'.format(hits_level))
            _write_data_to_file(hits_at_path, hits_value)

    # Write MRR to respective files.
    mr = np.mean(ranks)
    mrr = np.mean(1. / np.array(ranks))
    logging.info('Mean rank: %10.6f', mr)
    logging.info('Mean reciprocal rank: %10.6f', mrr)
    if enable_write_to_file:
        path_mr = os.path.join(results_dir, 'mean_rank.txt')
        path_mrr = os.path.join(results_dir, 'mrr.txt')
        _write_data_to_file(path_mr, mr)
        _write_data_to_file(path_mrr, mrr)
    logging.info('-' * 50)

    return mr, mrr, hits

def relation_extraction_metric_preprocessing(model, id2rel, data_iterator_handle,
                                name, session=None):

    predicted_rels = np.empty((0))
    correct_rels = np.empty((0))
    stopped = False
    while not stopped:
        try:
            e1, e2, rel, rel_multi, pred = session.run(
                (model.e1, model.e2, model.rel, model.rel_multi, model.predictions_all),
                feed_dict={model.input_iterator_handle: data_iterator_handle})
            pred[rel_multi == 0] = -np.inf
            predicted_rel = np.argmax(pred, axis=1)

            predicted_rel = np.array([id2rel[r] for r in predicted_rel])
            correct_rel = np.array([id2rel[r] for r in rel])

            correct_rels = np.concatenate((correct_rels, correct_rel), axis=0)
            predicted_rels = np.concatenate((predicted_rels, predicted_rel), axis=0)

        except tf.errors.OutOfRangeError:
            stopped = True

    return correct_rels, predicted_rels


def score(key, prediction, verbose=False):
    NO_RELATION = "no_relation"
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    metrics = {'precision': prec_micro, 'recall': recall_micro, 'f1': f1_micro}
    return metrics


if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
        parser.add_argument('gold_file', help='The gold relation file; one relation per line')
        parser.add_argument('pred_file',
                            help='A prediction file; one relation per line, in the same order as the gold file.')
        args = parser.parse_args()
        return args


    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (
        len(key), len(prediction)))
        exit(1)

    # Score the predictions
    score(key, prediction, verbose=True)
