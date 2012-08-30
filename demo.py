#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from sys import stdout

import numpy as np
from scipy import misc
from skdata import lfw

from sthor import model
from sthor.util import lru_cache
from bangreadout import LBFGSLogisticClassifier, ZScorer
from bangmetric import accuracy


def main():

    # -- dataset / splits
    lfw_view2 = lfw.view.AlignedView2()
    splits = lfw_view2.splits

    # -- Model
    print "SLM..."
    in_shape = (200, 200)
    # pre-defined model ("HT-L3-1st" from our FG11 paper):
    desc = model.parameters.fg11.fg11_ht_l3_1_description
    # random:
    #desc = model.parameters.plos09.get_random_plos09_l3_description()

    # using sthor:
    slm = model.slm.SequentialLayeredModel(in_shape, desc)
    # using thoreano:
    #from thoreano.slm import TheanoSLM
    #slm = TheanoSLM(in_shape, desc)
    #slm.transform = slm.process

    @lru_cache(maxsize=None)
    def process(arr_in):
        arr_in = misc.imresize(arr_in, in_shape).astype('float32')
        arr_in -= arr_in.min()
        arr_in /= arr_in.max()
        arr_out = slm.transform(arr_in)
        return arr_out

    def process_pair(pair):
        left = process(pair[0]).ravel()
        right = process(pair[1]).ravel()
        # -- comparison function
        diff = left - right
        absdiff = abs(diff)
        sqrtabsdiff = np.sqrt(absdiff)
        return sqrtabsdiff

    def process_pairs(pairs):
        print "Processing %d pairs..." % len(pairs)
        out = []
        for pi, pair in enumerate(pairs):
            if pi % 100 == 0:
                stdout.write('.%d' % (pi + 1))
                stdout.flush()
            farr = process_pair(pair).ravel()
            out += [farr]
        print
        return np.array(out)

    start = time.time()
    acc_l = []
    for split in splits:

        # -- train
        print '=' * 80
        print 'Training...'
        print '-' * 80
        X_trn_pairs = split.train.x
        y_trn = split.train.y
        y_trn = np.array(y_trn) > 0

        pair0 = X_trn_pairs[0]
        farr0 = process_pair(pair0).ravel()
        n_features = farr0.size

        clf = LBFGSLogisticClassifier(n_features=n_features)

        X_trn = process_pairs(X_trn_pairs)

        print 'z-scoring...'
        zscorer = ZScorer()
        X_trn = zscorer.fit(X_trn).transform(X_trn)

        print 'fitting...'
        clf.fit(X_trn, y_trn)
        # For screening, the training data will have to be further
        # divided for cross-validation, e.g. using StratifiedKFold from
        # sklearn.

        # -- test
        print '-' * 80
        print 'Testing...'
        print '-' * 80
        X_tst_pairs = split.test.x
        y_tst = split.test.y
        y_tst = np.array(y_tst) > 0

        X_tst = process_pairs(X_tst_pairs)

        print 'z-scoring...'
        X_tst = zscorer.transform(X_tst)

        print 'predicting...'
        y_pred = clf.predict(X_tst)

        acc = accuracy(y_tst, y_pred)
        print 'accuracy =', acc

        acc_l += [acc]

    end = time.time()

    print acc_l
    print 'average accuracy =', np.mean(acc_l)
    print 'time =', end - start


if __name__ == '__main__':
    main()
