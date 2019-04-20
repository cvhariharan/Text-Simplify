import nltk, gensim, operator, spacy, main_ppdb
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from enum import Enum
import json, thesaurus, math
import Conjugate


# use spacy for tagging

def generate_freq_dict():
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = {}
    for sentence in brown.sents():
        for word in sentence:
            word = word.lower()
            if word in freq_dict.keys():
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
    return freq_dict

class Simplify:

    def __init__(self, src):
        print("Initializing...")
        self.stop_words = set(stopwords.words('english')) 
        # freqFile = open("./freqs.json", 'r')
        self.freq = generate_freq_dict()
        self.freq_sum = sum(self.freq.values())
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300-SLIM.bin', binary=True)
        self.lemmatizer = WordNetLemmatizer()
        self.text = src
        self.ppdb = main_ppdb.load_ppdb(path='./ppdb-2.0-xxl-lexical', load_pickle=True)
        self.nlp = spacy.load('en_core_web_sm')
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

    def word2vec_similar_words(self, word, n=15):
        con = Conjugate.Conjugate()
        synonyms = []
        similar = []
        tag = self.nlp(word)[0].tag_
        print(tag)
        if tag != None:
            try:
                similar = self.model.most_similar(word, topn=n)
            except:
                pass
            for syn in similar:
                synConj = con.conjugate(syn[0], [word, tag])
                if synConj != None and synConj != word:
                    synonyms.append(syn)
        # print(synonyms)
        return synonyms

    def ppdb_words(self, word):
        if word in self.ppdb.keys():
            return self.ppdb[word]
        else:
            return []

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

    # Higher the better ( 0 - 1 )
    def getSimplicity(self, word):
        tag = self.nltk_tag_to_wordnet_tag(self.nlp(word)[0].tag_)
        if tag != None:
            wordLemma = self.lemmatizer.lemmatize(word, tag)
            if wordLemma != None:
                word = wordLemma
            if word in self.freq.keys():
                return self.freq[word]
        return 0

    def orderByComplexity(self, simIndex, originalWord):
        # Takes in the similarity tuples from getSynonyms and arranges in decreasing order of complexity * similarity
        index = {}
        for eachWord in simIndex:
            word = eachWord[0]
            score = eachWord[1]
            if word not in self.freq.keys():
                score = 0
            else:
                if self.freq[word] >= self.freq[originalWord]:
                    score = self.freq[word] * score
            index[word] = score
        return sorted(index.items(), key=operator.itemgetter(1), reverse=True)

    def getSynonyms(self, token):
        # returns a list of synonyms in decending order of similarity using word2vec as tuples with
        # word and similarity score
        con = Conjugate.Conjugate()
        tag = token.tag_
        word = token.text
        wordLemma = token.lemma_
        # print(wordLemma)

        # A very powerful spell
        index = {}
        try:
            # w = thesaurus.Word(word)
            for eachWord in self.synonimize(word, tag):
                eachWord = con.conjugate(eachWord, [word, tag])
                if eachWord != word and eachWord is not None:
                    index[eachWord] = self.model.similarity(word, eachWord)
        except:
            pass
        return sorted(index.items(), key=operator.itemgetter(1), reverse=True)

    def check_if_fits(self, sentence, word, replacement, threshold=0.1):
        newSent = sentence.replace(word, replacement)
        return self.model.wmdistance(newSent, sentence) < threshold

    def testContext(self, sentence, word, replacements):
        # uses WMD to test sentence similarity after replacing
        replacements = replacements[:3]
        token = self.nlp(word)[0]
        tag = token.tag_
        # word = token.text
        smallest = math.inf
        bestSent = ""
        con = Conjugate.Conjugate()
        for eachReplacement in replacements:
            # splitSent = sentence.split(" ")
            replacement = eachReplacement[0]
            replacement = con.conjugate(replacement, [word, tag])
            if replacement != word:
                # Should not be doing this but this spell is tempting
                try:
                    # splitSent[wordIndex] = eachReplacement[0]
                    newSent = sentence.replace(word, replacement) #" ".join(splitSent) 
                    if self.model.wmdistance(newSent, sentence) < smallest:
                        smallest = self.model.wmdistance(newSent, sentence)
                        bestSent = newSent
                except:
                    pass
        
        if bestSent != "":
            return bestSent
        return sentence

    def getWeightedScore(self, a, b, alpha=0.7, beta=0.3):
        # return 1/((alpha/(a+1)) + (beta)/(b+1))
        return (alpha * a + beta * b) / 2

    def simplify(self):
        replaced = set()
        tokens = self.nlp(self.text)
        text = self.text
        i = 0
        print(sum(self.freq.values()))
        top_n = 3000
        freq_top_n = sorted(self.freq.values(), reverse=True)[top_n - 1]
        # print(freq_top_n)
        difficult = []
        index = {}
        for t in tokens:
            if self.nltk_tag_to_wordnet_tag(t.tag_) != None and self.isReplaceable(t.text):
                wordLemma = self.lemmatizer.lemmatize(t.text, self.nltk_tag_to_wordnet_tag(t.tag_))
                if wordLemma != None:
                    # print(wordLemma)
                    if wordLemma in self.freq.keys():
                        if self.freq[wordLemma] < freq_top_n:
                            difficult.append(t.text)
                    else:
                        difficult.append(t.text)
                    # print(t.text + " - > " + str(self.ppdb_words(t.text)))
        print(difficult)
        optionsDict = {}
        for dif in difficult:
            options = {}
            # opclear
            # tionsDict['ppdb'] = []
            con = Conjugate.Conjugate()
            # synonyms = self.word2vec_similar_words(dif)
            # print(synonyms)
            for synonym in self.word2vec_similar_words(dif):
                word = synonym[0]
                similarity = synonym[1]
                complexity = self.getSimplicity(word)
                options[word] = self.getWeightedScore(complexity, similarity, 0.8, 0.2)

            optionsDict[dif] = sorted(options.items(), key=operator.itemgetter(1), reverse=True)
        print(optionsDict)
        



src = "Nevertheless, they spoke with a common paradigm in mind; they shared the Marxist Hegelian mindset and were preoccupied with similar questions."
# src = "The river is formed through the confluence of the Macintyre River and Weir River (part of the Border Rivers system), north of Mungindi, in the Southern Downs region of Queensland. The Barwon River generally flows south and west, joined by 36 tributaries, including major inflows from the Boomi, Moonie, Gwydir, Mehi, Namoi, Macquarie, Bokhara and Bogan rivers. During major flooding, overflow from the Narran Lakes and the Narran River also flows into the Barwon. The confluence of the Barwon and Culgoa rivers, between Brewarrina and Bourke, marks the start of the Darling River."
# src = "It was believed that illnesses were brought on humans by demons and these beliefs and rituals could have prehistoric roots. According to folklore, the 18 demons who are depicted in the Sanni Yakuma originated during the time of the Buddha.[N 1] The story goes that the king of Licchavis of Vaishali suspected his queen of committing adultery and had her killed. However, she gave birth when she was executed and her child became the Kola Sanniya, who grew up 'feeding on his mother's corpse'. The Kola Sanni demon destroyed the city, seeking vengeance on his father, the king. He created eighteen lumps of poison and charmed them, thereby turning them into demons who assisted him in his destruction of the city.They killed the king, and continued to wreak havoc in the city, 'killing and eating thousands' daily, until finally being tamed by the Buddha and agreed to stop harming humans."
# src = "The enemy captured many soldiers. The captured army soldier, after waiting, secretly captured pictures that captured the war zone. Her invention will capture carbon dioxide and will capture the silver medal. She captured her friend's chess pieces and then, after capturing more footage, she captured our hearts. Her friend made a mint while chewing a mint. He tried to mint quarters and sell these quarters to the Philadelphia Mint. He liked mint tea and studying mint leaves while working on Linux Mint. But, the breath mints he always mints don't taste like mint chip ice cream and don't earn him mints."
s = Simplify(src.lower())
# nlp = spacy.load('en_core_web_sm')
# w = nltk.pos_tag(['paradigms'])
# c = Conjugate.Conjugate()
# print(c.conjugate("fundamentally", ['paradigm', nlp('paradigm')[0].tag_]))
