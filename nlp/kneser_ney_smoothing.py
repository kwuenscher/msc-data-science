from numpy.random import permutation
import collections
from collections import Counter


# Bigram base class
class BiGram(lm.LanguageModel):

    def __init__(self, train_set, order):

        super().__init__(set(train_set), order)

        self.train_set = train_set
        self.unseen_words = []

        bigrams = list(zip(train_set, train_set[1:]))
        unigrams = list(zip(train_set))

        self.bi_num = Counter(bigrams)
        self.uni_num = Counter(unigrams)

        self.bi_dnom = Counter(np.array(bigrams)[:, 0])
        self.uni_dnom = len(self.train_set)

    def updateVocab(self, vocab):
        self.unseen_words = vocab - self.vocab

# Implementation of basic Kneser-Ney smoothing for a bigram model.
# This is the best performing model I have created and hence I am using this class for the assignment.

class KneNeySm(BiGram):

    def __init__(self, train_set, order, discount = 0.5):
        super().__init__(train_set, order)

        self.discount = discount
        self.oov = "[OOV]"
        self.unseen_words = []

        self.unseen_context = collections.defaultdict(float)
        self.totalBiTypes = 0
        self.bi_w_follows = collections.defaultdict(list)
        self.prefix_counter = collections.defaultdict(float)
        self.bi_context_counter = collections.defaultdict(float)

        self.initOrderCounts()

    def initOrderCounts(self):

        # Temporarily storing tuples of dictonary in ordered list.
        bi_buffer = np.array([key for key in self.bi_num])

        self.totalBiTypes = len(self.bi_num)

        for i in range(0, len(bi_buffer)):
            self.prefix_counter[bi_buffer[i, 1]] += 1
            self.bi_w_follows[bi_buffer[i, 0]].append(bi_buffer[i, 1])
            self.bi_context_counter[bi_buffer[i, 0]] += 1

        # Returns discounted probability of a bigram
    def getBiProb(self, word, context):

        return max(self.bi_num[tuple(context + tuple([word]))] - self.discount, 0) / self.bi_dnom[context[0]]

        # Returns continuation probability
    def getContiProb(self, word):

        return self.prefix_counter[word]/ self.totalBiTypes

        # Extracts the number of contexts seen for a specific word as well as all words that have been
        # seen following that word.
    def findCount(self, context):

        return self.bi_context_counter[context[0]], self.bi_w_follows[context[0]]

        # Returns normalising contstant for backoff weight calculation.
    def getSmoothDnom(self, words, context):
        prob = []
        for word in words:
            prob.append(self.getContiProb(word))

        return 1 - sum(prob)

        # Returns smooth probabiltiy for novel context/word combinations.
    def smoothContext(self, word, context):
        count, words = self.findCount(context)
        reserved_mass = count * self.discount / self.bi_dnom[context[0]]
        norm = self.getSmoothDnom(words, context)
        backoff_prob = reserved_mass / norm
        self.unseen_context[context] = backoff_prob

        return backoff_prob * self.getContiProb(word)

        # Updating the discount factor.
    def updateDisc(self, discount):
        # Old context dict needs to be flushed.
        self.unseen_context = collections.defaultdict(float)
        self.discount = discount

        # Returns the ML of a seen word within a context.
    def probability(self, word, *context):

        if word not in self.vocab:
            return self.probability(self.oov, *context) / len(self.unseen_words)

        elif context[0] not in self.bi_dnom:
            return self.getContiProb(word)

        elif context in self.unseen_context:

            if tuple(context + tuple([word])) not in self.bi_num:
                return self.unseen_context[context] * self.getContiProb(word)
            else:
                return self.getBiProb(word, context)

        elif tuple(context + tuple([word])) not in self.bi_num:

            return self.smoothContext(word, context)

        else:
            return self.getBiProb(word, context)
