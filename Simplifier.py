import nltk, gensim, operator
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from enum import Enum
import json, thesaurus, math


class Simplify:

    def __init__(self, src):
        print("Initializing...")
        self.stop_words = set(stopwords.words('english')) 
        freqFile = open("./freqs.json", 'r')
        self.freq = json.load(freqFile)
        self.model = gensim.models.KeyedVectors.load('./word2vec-norm', mmap='r')
        self.lemmatizer = WordNetLemmatizer()
        self.text = src
        print("Complete.")
        self.simplify()

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

    def conjugate(self, originalWord, synonym):
        # Returns the synonym in correct tense and plurality

    def synonimize(self, word, pos=None):
        # Get synonyms of the word / lemma
        try:
            # map part of speech tags to wordnet
            pos = {'NN': wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}[pos[:2]]
            # print(pos)
        except:
            return [word]

        synsets = wordnet.synsets(word, pos)
        synonyms = []
        for synset in synsets:
            for sim in  synset.similar_tos():
                synonyms += sim.lemma_names()

        # return list of synonyms or just the original word
        return synonyms or [word]

    def orderByComplexity(self, simIndex):
        # Takes in the similarity tuples from getSynonyms and arranges in decreasing order of complexity * similarity
        index = {}
        for eachWord in simIndex:
            word = eachWord[0]
            score = eachWord[1]
            if word not in self.freq.keys():
                score = math.inf
            else:
                score = self.freq[word] * score
            index[word] = score
        return sorted(index.items(), key=operator.itemgetter(1), reverse=True)

    def getSynonyms(self, word):
        # returns a list of synonyms in decending order of similarity using word2vec as tuples with
        # word and similarity score
        tag = nltk.pos_tag([word])[0][1]
        wordLemma = self.lemmatizer.lemmatize(word, self.nltk_tag_to_wordnet_tag(tag))
        # print(wordLemma)
        # w = thesaurus.Word(word)
        index = {}

        # A very powerful spell
        try:
            for eachWord in self.synonimize(wordLemma, tag):
                if eachWord != word and eachWord is not None:
                    index[eachWord] = self.model.similarity(word, eachWord)
        except:
            pass
        return sorted(index.items(), key=operator.itemgetter(1), reverse=True)

    def testContext(self, sentence, word, replacements):
        # uses WMD to test sentence similarity after replacing
        for eachReplacement in replacements:
            newSent = sentence.replace(word, eachReplacement[0])
            print(eachReplacement[0] + " : " + str(self.model.wmdistance(newSent, sentence)))

    def simplify(self):
        sentTokens = sent_tokenize(self.text)
        for sent in sentTokens:
            wordsList = nltk.word_tokenize(sent)
            wordsList = [w for w in wordsList if not w in self.stop_words]
            for word in wordsList:
                if self.isReplaceable(word):
                    print(self.orderByComplexity(self.getSynonyms(word)))


src = "The river is formed through the confluence of the Macintyre River and Weir River (part of the Border Rivers system), north of Mungindi, in the Southern Downs region of Queensland. The Barwon River generally flows south and west, joined by 36 tributaries, including major inflows from the Boomi, Moonie, Gwydir, Mehi, Namoi, Macquarie, Bokhara and Bogan rivers. During major flooding, overflow from the Narran Lakes and the Narran River also flows into the Barwon. The confluence of the Barwon and Culgoa rivers, between Brewarrina and Bourke, marks the start of the Darling River."
s = Simplify(src.lower())
