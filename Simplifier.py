import nltk, gensim, operator, spacy
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from enum import Enum
import json, thesaurus, math
import Conjugate, pprint
import pandas as pd

class Simplify:

    def __init__(self, src):
        print("Initializing...")
        self.stop_words = set(stopwords.words('english')) 
        # freqFile = open("./freqs.json", 'r')
        self.freq = self.getFreqDict()
        self.freq_sum = sum(self.freq.values())
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300-SLIM.bin', binary=True)
        self.lemmatizer = WordNetLemmatizer()
        self.text = src
        self.pp = pprint.PrettyPrinter(indent=4)

        ngrams = pd.read_csv('ngrams.csv')
        ngrams = ngrams.drop_duplicates(subset='bigram', keep='first')
        self.ngram_freq = dict(zip(ngrams.bigram, ngrams.freq))

        self.nlp = spacy.load('en_core_web_sm')
        print("Complete.")
        self.simplify()

    def getFreqDict(self):
        # Uses brown corpus
        freq_dict = {}
        for sentence in brown.sents():
            for word in sentence:
                word = word.lower()
                if word in freq_dict.keys():
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1
        return freq_dict

    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def isReplaceable(self, word):
        # Only replace nouns, adjectives and verbs
        tag = self.nltk_tag_to_wordnet_tag(nltk.pos_tag([word])[0][1])
        return ( tag == wordnet.ADJ or tag == wordnet.VERB or tag == wordnet.NOUN )

    def word2vec_similar_words(self, word, n=15):
        # Returns the most similar words using word2vec and conjugates the words
        con = Conjugate.Conjugate()
        synonyms = []
        similar = []
        tag = self.nlp(word)[0].tag_
        if tag != None:
            try:
                similar = self.model.most_similar(word, topn=n)
            except:
                pass
            for syn in similar:
                synConj = con.conjugate(syn[0], [word, tag])
                if synConj != None and synConj != word:
                    synonyms.append(syn)
        return synonyms

    
    def getSimplicity(self, word):
        # Higher the better, not normalized
        tag = self.nltk_tag_to_wordnet_tag(self.nlp(word)[0].tag_)
        if tag != None:
            wordLemma = self.lemmatizer.lemmatize(word, tag)
            if wordLemma != None:
                word = wordLemma
            if word in self.freq.keys():
                return self.freq[word]
        return 0

    def checkIfFits(self, word, context):
        # Checks if bigram with this context exist
        key1 = context[0] + " " + word
        key2 = word + " " + context[1]
        return key1 in self.ngram_freq.keys() or key2 in self.ngram_freq.keys()

    def getBigramCount(self, word, context):
        # Returns the bigram frequency from the dataset
        key1 = context[0] + " " + word
        key2 = word + " " + context[1]
        score = 0
        if key1 in self.ngram_freq.keys():
            score += self.ngram_freq[key1]
        if key2 in self.ngram_freq.keys():
            score += self.ngram_freq[key2]
        return score

    def getARI(self, text):
        # Returns the automated readability index of a text
        characters = len(text)
        wordCount = len(text.split(" "))
        sentences = len(sent_tokenize(text))

        return (4.71*(characters/wordCount) + 0.5*(wordCount/sentences) - 21.43)

    def getWeightedScore(self, a, b, c, alpha=0.7, beta=0.3, gamma=0.5):
        # Returns a weighted average
        # Could be made more elegant but I am too lazy and tired after spending hours on this project
        # return 1/((alpha/(a+1)) + (beta)/(b+1))
        return (alpha * a + beta * b + gamma * c) / 3

    def simplify(self):
        replaced = set()
        tokens = self.nlp(self.text)
        text = self.text
        i = 0

        # print(sum(self.freq.values()))
        top_n = 3000
        freq_top_n = sorted(self.freq.values(), reverse=True)[top_n - 1]

        # dict that stores each difficult word against its context
        difficult = {}
        index = {}
        count = 0
        for t in tokens:
            if self.nltk_tag_to_wordnet_tag(t.tag_) != None and self.isReplaceable(t.text):
                wordLemma = self.lemmatizer.lemmatize(t.text, self.nltk_tag_to_wordnet_tag(t.tag_))
                if wordLemma != None:
                    
                    if wordLemma in self.freq.keys():
                        if self.freq[wordLemma] < freq_top_n:
                            difficult[t.text] = [tokens[count-1].text, tokens[count+1].text]
                    else:
                        difficult[t.text] = [tokens[count-1].text, tokens[count+1].text]
            count += 1

        self.pp.pprint(difficult)
        optionsDict = {}
        for dif in difficult.keys():
            options = {}
            con = Conjugate.Conjugate()
            for synonym in self.word2vec_similar_words(dif):
                word = synonym[0]
                similarity = synonym[1]
                complexity = self.getSimplicity(word)
                bigramScore = self.getBigramCount(word, difficult[dif])
                options[word] = self.getWeightedScore(complexity, similarity, bigramScore, 0.6, 0.4, 1)

            optionsDict[dif] = sorted(options.items(), key=operator.itemgetter(1), reverse=True)
        self.pp.pprint(optionsDict)


        sentences = sent_tokenize(self.text)
        finalText = ""
        for sent in sentences:
            sentScore = self.getARI(sent)
            for dif in optionsDict.keys():
                if dif in sent:
                    m = 0
                    optionsLen = len(optionsDict[dif])
                    
                    if optionsLen > 0:
                        while not self.checkIfFits(optionsDict[dif][m][0], difficult[dif]):
                            m += 1
                            if m > optionsLen-1:
                                break
                        if m > 0:
                            m = m - 1
                        if self.checkIfFits(optionsDict[dif][m][0], difficult[dif]):
                            # check if improves readability
                            newSentScore = self.getARI(sent.replace(dif, optionsDict[dif][m][0]))
                            if newSentScore < sentScore:
                                sent = sent.replace(dif, optionsDict[dif][m][0])
            finalText = finalText + sent + "\n"
        
        print(finalText)

src = "Yet, Sherman’s bedfellows are far from strange. Art, despite its religious and magical origins, very soon became a commercial venture. From bourgeois patrons funding art they barely understood in order to share their protegee’s prestige, to museum curators stage-managing the cult of artists in order to enhance the market value of museum holdings, entrepreneurs have found validation and profit in big-name art. Speculators, thieves, and promoters long ago created and fed a market where cultural icons could be traded like commodities."
s = Simplify(src.lower())
