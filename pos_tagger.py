import glob
import re
from collections import Counter
from nltk.corpus.reader import TaggedCorpusReader
from collections import defaultdict


class pos_tagger():

    def __init__(self):
        self.unknown_prob = 0.0000000000001
        self.tagged_file = glob.glob("brown/*")
        self.bigram_cnt = {}
        self.unigram_cnt = {}
        self.tag_count = defaultdict(lambda: 0)
        self.tag_word_count = Counter()
        self.transition_probabilities = defaultdict(lambda: self.unknown_prob)
        self.emmission_probabilities = defaultdict(lambda: self.unknown_prob)

    def ngrams(self, text, n):
        Ngrams = []
        for i in range(len(text)): Ngrams.append(tuple(text[i: i + n]))
        return Ngrams

    def bigram_counts(self, tags):
        for i_tag_bigram in self.ngrams(tags, 2):
            if i_tag_bigram in self.bigram_cnt:
                self.bigram_cnt[i_tag_bigram] += 1
            else:
                self.bigram_cnt[i_tag_bigram] = 1
        return self.bigram_cnt

    def unigram_counts(self, tags):
        for tag in tags:
            if tag in self.unigram_cnt:
                self.unigram_cnt[tag] += 1
            else:
                self.unigram_cnt[tag] = 1
        return self.unigram_cnt

    def tag_word_counts(self, tagged_words):
        for tag, word in tagged_words:
            self.tag_count[tag] += 1
            if (word, tag) in self.tag_word_count:
                self.tag_word_count[(tag, word)] += 1
            else:
                self.tag_word_count[(tag, word)] = 1
        return self.tag_word_count

    def transition_probabilty(self, tags):
        bigrams = self.ngrams(tags, 2)
        for bigram in bigrams:
            self.transition_probabilities[bigram] = self.bigram_cnt[bigram] / self.unigram_cnt[bigram[0]]
        return self.transition_probabilities

    def emmission_probabilty(self, tagged_words):
        for tag, word in tagged_words:
            self.emmission_probabilities[tag, word] = self.tag_word_count[tag, word] / self.tag_count[tag]
        return self.emmission_probabilities

    def initial_probabilities(self, tag):
        return self.transition_probabilities["START", tag]

    def vertibi(self, observable, in_states):
        states = set(in_states)
        states.remove("START")
        states.remove("END")
        trails = {}
        for s in states:
            trails[s, 0] = self.initial_probabilities(s) * self.emmission_probabilities[s, observable[0]]
        for o in range(1, len(observable)):
            obs = observable[o]
            for s in states:
                v1 = [(trails[k, o - 1] * self.transition_probabilities[k, s] * self.emmission_probabilities[s, obs], k) for k in states]
                k = sorted(v1)[-1][1]
                trails[s, o] = trails[k, o - 1] * self.transition_probabilities[k, s] * self.emmission_probabilities[s, obs]
        best_path = []
        for o in range(len(observable) - 1, -1, -1):
            k = sorted([(trails[k, o], k) for k in states])[-1][1]
            best_path.append((observable[o], k))
        best_path.reverse()
        for x in best_path:
            print(str(x[0]) + "," + str(x[1]))
        return best_path

    def clean(self, word):
        word = re.sub('\s+', '', word.lower())
        return word

    def tag_test(self, all_tags):
        words = []
        with open("tag_test.txt") as f:
            for line in f:
                if "sentence ID" in line:
                    words = []
                    print(line)
                elif "<EOS>" in line:
                    self.vertibi([self.clean(w) for w in words], all_tags)
                    print("<EOS>")
                else:
                    words.append(line)

    def tag(self):
        reader_corpus = TaggedCorpusReader('.',
                                           self.tagged_file)

        tagged_words = []
        all_tags = []
        for sent in reader_corpus.tagged_sents():  # get tagged sentences
            all_tags.append("START")
            for (word, tag) in sent:
                if tag is None or tag in ['NIL']:
                    continue
                all_tags.append(tag)
                word = self.clean(word)
                tagged_words.append((tag, word))
            all_tags.append("END")

        self.tag_word_counts(tagged_words)

        self.bigram_cnt = self.bigram_counts(all_tags)
        self.unigram_cnt = self.unigram_counts(all_tags)

        self.transition_probabilty(all_tags)
        self.emmission_probabilty(tagged_words)

        self.tag_test(all_tags)


ps = pos_tagger()
print(ps.tag())
