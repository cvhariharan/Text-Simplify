import pattern.en, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pattern.en import PARTICIPLE

class Conjugate:
    def __init__(self):
        # self.word = word
        # self.word_pos = nltk.pos_tag([word])[0]
        self.lemmatizer = WordNetLemmatizer()

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

    def isPlural(self, word):
        wordLemma = self.lemmatizer.lemmatize(self.word, self.nltk_tag_to_wordnet_tag(self.word_pos[1]))
        return not (wordLemma == word)

    def conjugate(self, synonym, word_pos):
        # check if noun
        # print(word_pos[1])
        if self.nltk_tag_to_wordnet_tag(word_pos[1]) == wordnet.NOUN:
            # check if synonym and word have same plurality, if not conjugate
            if self.isPlural(word_pos[0]):
                return pattern.en.pluralize(synonym)
            else:
                return pattern.en.singularize(synonym)
        
        # check if adjective and return proper form
        if self.nltk_tag_to_wordnet_tag(word_pos[1]) == wordnet.ADJ:
            if len(word_pos[1]) == 2:
                #positive
                return synonym
            if word_pos[1][-1] == 'R':
                #comparative
                return pattern.en.comparative(synonym)
            if word_pos[1][-1] == 'S':
                # superlative
                return pattern.en.superlative(synonym)
        
        # check if verb and return proper tense
        if self.nltk_tag_to_wordnet_tag(word_pos[1]) == wordnet.VERB:
            if word_pos[1] == 'VBD':
                return pattern.en.conjugate(synonym, 'past')
            if word_pos[1] == 'VBN':
                return pattern.en.conjugate(synonym, 'past')
            if word_pos[1] == 'VBZ':
                return pattern.en.conjugate(synonym, 'present', '3sg')
            if word_pos[1] == 'VBP':
                return pattern.en.conjugate(synonym, 'present', '1sg')
            if word_pos[1] == 'VBG':
                return pattern.en.conjugate(synonym, tense=PARTICIPLE, parse=True)
            if word_pos[1] == 'VB':
                return pattern.en.conjugate(synonym, 'infinitive')


