import math
import nltk
import time
from nltk.tokenize import RegexpTokenizer

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):

    unigrams, bigrams, trigrams = {}, {}, {}

    for s in training_corpus:
        tokens = s.strip().split() + [STOP_SYMBOL]
        for w in tokens:
            if w in unigrams:
                unigrams[w] += 1
            else:
                unigrams[w] = 1

        tokens = [START_SYMBOL] + tokens
        bigram_tuples = tuple(nltk.bigrams(tokens))
        for b in bigram_tuples:
            if b in bigrams:
                bigrams[b] += 1
            else:
                bigrams[b] = 1

        tokens = [START_SYMBOL] + tokens
        trigram_tuples = tuple(nltk.trigrams(tokens))
        for t in trigram_tuples:
            if t in trigrams:
                trigrams[t] += 1
            else:
                trigrams[t] = 1

    wc = sum(unigrams.itervalues())
    unigram_p = {z: math.log(float(c) / wc, 2) for z, c in unigrams.iteritems()}

    unigrams[START_SYMBOL] = len(training_corpus)
    bigram_p = {z: math.log(float(c) / unigrams[z[0]], 2) for z, c in bigrams.iteritems()}

    bigrams[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {z: math.log(float(c) / bigrams[z[:2]], 2) for z, c in trigrams.iteritems()}

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for s in corpus:
        x = 0
        t = s.strip().split()
        if n == 1: tks = t + [STOP_SYMBOL]
        elif n == 2: tks = nltk.bigrams([START_SYMBOL] + t + [STOP_SYMBOL])
        elif n == 3: tks = nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + t + [STOP_SYMBOL])
        else: raise ValueError('error %s' % n)
        for t in tks:
            try:
                v = ngram_p[t]
            except KeyError:
                v = MINUS_INFINITY_SENTENCE_LOG_PROB
            x += v
        scores.append(x)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    for s in corpus:
        interpolated = 0.0
        tokens = s.strip().split()
        tokens.insert(0, START_SYMBOL)
        tokens.insert(0, START_SYMBOL)
        tokens.append(STOP_SYMBOL)
        sent_trigrams = tuple(nltk.trigrams(tokens))

        for t in sent_trigrams:
            u = (t[2],)
            b = (t[1], t[2],)
            if t not in trigrams or b not in bigrams or u not in unigrams:
                interpolated = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
                interpolated += math.log(((1 / 3.0) * 2 ** trigrams[t]) +
                                         ((1 / 3.0) * 2 ** bigrams[b]) +
                                         ((1 / 3.0) * 2 ** unigrams[u]), 2)

        scores.append(interpolated)

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
