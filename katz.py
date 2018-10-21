#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
katz.py
Desc :  Katz algorithm used to predict circRNA-disease associations.
Usage: ./katz.py directory-contains-feature-files outDir randomSeed
E.g. : ./katz.py ./features ./katz-outdir 1
Coder: linwei, etc
Created date: 20180305
'''
import time
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from scipy import stats

import sys
from os import listdir
from os.path import isfile, join

if(len(sys.argv) != 4):
    sys.exit("Usage: %s directory-for-input-files outdir randomSeed\n" %(sys.argv[0]))

def calPRAUC(ranks, nTPs, topN):
    cumPRAUC = 0
    posRecalls = list()
    for i in range(topN):
        if(ranks[i] < nTPs):
            posRecalls.append(1)
        else:
            posRecalls.append(0)

    curSum = posRecalls[0]
    prevRecall = round(posRecalls[0] / nTPs, 4)
    prevPrec = round(posRecalls[0], 4)
    for i in range(1, topN):
        curSum += posRecalls[i]
        recall = round(curSum / nTPs, 4)
        prec   = round(curSum / (i+1), 4)
        cumPRAUC += ((recall - prevRecall) * (prevPrec + prec) / 2)
        prevRecall = recall
        prevPrec = prec

    cumPRAUC = round(cumPRAUC, 4)
    return cumPRAUC

inpath = sys.argv[1]
outdir = sys.argv[2]
rs = int(sys.argv[3])

nfolds = 5
topN = 500
powers = [1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]  #damping factors for increasing length
beltas = np.power(2.0, range(-7, 0))
nBelta = beltas.shape[0]
dFeatures = [f for f in listdir(inpath) if isfile(join(inpath, f))]
for df in dFeatures:
    print("Processing %s" %(df) )
    dId = df.split('_')[0]
    pf = "/".join((inpath, df)) #processing input file: df

    outfile = ".".join((dId, "txt"))
    of = "/".join((outdir, outfile))

    eRecalls = np.zeros(nfolds)
    ePrecisions = np.zeros(nfolds)
    ePRAUCs = np.zeros(nfolds)

    d = np.loadtxt(pf, delimiter = ',')
    p = d[d[:, 24] == 1, :]
    u = d[d[:, 24] == 0, :]
    X_p = p[:, 0:12]  #Only the first 12 features(path-count) are used in the KATZ algorithm
    y_p = p[:, 24]
    X_u = u[:, 0:12]
    y_u = u[:, 24]
        
    #nfolds to evaluate the performance
    ikf = 0
    kf = KFold(n_splits = nfolds, shuffle = True, random_state = rs)
    X_p_splits = list(kf.split(X_p))
    X_u_splits = list(kf.split(X_u))
    for ikf in range(nfolds):
        p_train_index, p_test_index = X_p_splits[ikf]
        u_train_index, u_test_index = X_u_splits[ikf]
        X_p_train = X_p[p_train_index]
        y_p_train = y_p[p_train_index]
        X_p_test  = X_p[p_test_index]
        y_p_test  = y_p[p_test_index]
        
        X_u_train= X_u[u_train_index]
        y_u_train= y_u[u_train_index]
        X_u_test = X_u[u_test_index]
        y_u_test = y_u[u_test_index]
#        print("Train:", X_p_train.shape, "test:", X_p_test.shape)
        start_time = time.time()
        cvMeans = np.zeros(nBelta)
        cvStds = np.zeros(nBelta)
        ithPair = 0
        #nested nfolds to select optimal parameters
        kf2 = KFold(n_splits = nfolds, shuffle = True, random_state = rs)
        for belta in beltas:
            recalls = np.zeros(nfolds)
            X_p_cv_splits = list(kf2.split(X_p_train))
            X_u_cv_splits = list(kf2.split(X_u_train))
            for ikf2 in range(nfolds):
                p_train_cv_index, p_val_cv_index = X_p_cv_splits[ikf2]
                u_train_cv_index, u_val_cv_index = X_u_cv_splits[ikf2]
                X_p_cv_train = X_p_train[p_train_cv_index]
                y_p_cv_train = y_p_train[p_train_cv_index]
                X_p_cv_val   = X_p_train[p_val_cv_index]
                y_p_cv_val   = y_p_train[p_val_cv_index]
                
                X_u_cv_train = X_u_train[u_train_cv_index]
                y_u_cv_train = y_u_train[u_train_cv_index]
                X_u_cv_val = X_u_train[u_val_cv_index]
                y_u_cv_val = y_u_train[u_val_cv_index]
                #mix validation + unlabel for transductive learning to see how it perform on validation set
                X_pu_cv_val = np.concatenate((X_p_cv_val, X_u_cv_val), axis = 0)
                y_pu_cv_val = np.concatenate((y_p_cv_val,  y_u_cv_val), axis = 0)
                #scores = belta * pA21 + belta^2 * (pA31 + pA32 +  pA33) + belta^3 * (pA41 + pA42 + A43 + pA44 + pA45 + pA46 + pA47 + pA48)
                weights = np.power(belta, powers)
                s = weights * X_pu_cv_val
                #print("Before weighted:", X_p_cv_val[0])
                #print("After weighted:", s[0])
                scores = np.sum(s, axis = 1)
                #print("scores.shape:", scores.shape)
                orderScores = np.argsort(-scores) #sort in descent order
                topNIndex = orderScores[:topN]
                truePosIndex = np.array(range(y_p_cv_val.shape[0]) ) #they are the firstN rows in the concatenated validation set
                truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)
                recall = truePosRecall.shape[0] / truePosIndex.shape[0]
                #recall = calPRAUC(orderScores, y_p_cv_val.shape[0], topN)
                recalls[ikf2] = recall
            avgRecall = np.mean(recalls)
            cvMeans[ithPair] = avgRecall
            stdRecall = np.std(recalls)
            cvStds[ithPair] = stdRecall
            #print("For belta: %f, rank of top %d: average recall: %.2f%%, std of recall: %.2f" %(belta, topN, avgRecall*100, stdRecall ))
            #print("For each fold:", recalls)
            ithPair += 1
        elapsed_time = time.time() - start_time
        cvMaxMeanIndex = np.argmax(cvMeans)
        optimalBelta = beltas[cvMaxMeanIndex % nBelta]
        #print("cv-MaxMean:", cvMeans[cvMaxMeanIndex], "cv-MaxMean_std:", cvStds[cvMaxMeanIndex], "cvMaxMeanIndex:", cvMaxMeanIndex)
        print("disease:", dId, ", randomSeed:", rs, "ithFold:", ikf, ", optimalBelta:", optimalBelta, ", cv-MaxMean:", cvMeans[cvMaxMeanIndex])
        #print("cross-validation time elapsed: %.2f" %(elapsed_time) )
        #After parameter selection, we evaluate on the test set with the optimal parameters
        X_test = np.concatenate((X_p_test, X_u_test), axis = 0)
        y_test = np.concatenate((y_p_test,  y_u_test), axis = 0)
        weights = np.power(optimalBelta, powers) #Modified@20180424, add, didn't update the weights before!!!
        s = weights * X_test
        #print("Before weighted:", X_p_cv_val[0])
        #print("After weighted:", s[0])
        scores = np.sum(s, axis = 1)
        #print("scores.shape:", scores.shape)
        orderScores = np.argsort(-scores) #sort in descent order
        topNIndex = orderScores[:topN]      
        #print("topN scores index:", topNIndex)
        truePosIndex = np.array(range(y_p_test.shape[0]) ) 
        truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)

        orderList = [str(item) for item in orderScores]
        orderStr = ','.join(orderList)
        recall = truePosRecall.shape[0] / truePosIndex.shape[0]
        precision = truePosRecall.shape[0] / topN
        prauc = calPRAUC(orderScores, y_p_test.shape[0], topN)
        eRecalls[ikf] = recall
        ePrecisions[ikf] = precision
        ePRAUCs[ikf] = prauc
        print("dId: %s, randomState: %d, %dth-fold, recall: %.2f%%, precision: %.2f%%, prauc: %.4f" %(dId, rs, ikf, recall*100, precision*100, prauc))
        #print("For disease: %s, %d test fold, rank of top %d: recall: %.2f%%, precision: %.2f%%" %(dId, ikf, topN, recall*100, precision*100))
        with open(of, "a") as output:
            output.write("%s-RandomState%d-%dth fold, number of true positive:%d\n" %(dId, rs, ikf, y_p_test.shape[0]))
            output.write("%s\n" %(orderStr))
            output.write("END\n")
    mRecall = np.mean(eRecalls)
    stdRecall = np.std(eRecalls)
    mPrec   = np.mean(ePrecisions)
    stdPrec = np.std(ePrecisions)
    mPRAUC = np.mean(ePRAUCs)
    stdPRAUC = np.std(ePRAUCs)
    recallList = [str(item) for item in eRecalls]
    precList   = [str(item) for item in ePrecisions]
    praucList  = [str(item) for item in ePRAUCs]
    recallStr = ','.join(recallList)
    precStr = ','.join(precList)
    praucStr = ','.join(praucList)
    with open (of, "a") as output:
        output.write("%s-RandomState%d, mean+-std recall:%.4f,%.4f\n" %(dId, rs, mRecall, stdRecall))
        output.write("%s-RandomState%d, mean+-std precision:%.4f,%.4f\n" %(dId, rs, mPrec, stdPrec))
        output.write("%s-RandomState%d, mean+-std prauc:%.4f,%.4f\n" %(dId, rs, mPRAUC, stdPRAUC))
        output.write("%s-RandomState%d, 5-fold cv recall:%s\n" %(dId, rs, recallStr))
        output.write("%s-RandomState%d, 5-fold cv precision:%s\n" %(dId, rs, precStr))
        output.write("%s-RandomState%d, 5-fold cv prauc:%s\n" %(dId, rs, praucStr))
        output.write("END\n")
    print("summary of %s, randomSeed: %d, top %d, mean/std of prauc, mean/std of recall, mean/std of precision: %f,%f,%f,%f,%f,%f" %(dId, rs, topN, mPRAUC, stdPRAUC, mRecall, stdRecall, mPrec, stdPrec))
    print(eRecalls)
    print(ePrecisions)
    print(ePRAUCs)
    print("END")
