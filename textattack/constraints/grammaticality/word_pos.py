"""
------------------------------
"""
import nltk
from textattack.constraints import Constraint


class POSCheck(Constraint):

    def __init__(self, allow_pos_pref=["ADJ", "ADV", "VERB", "NOUN"], compare_against_original=True):
        super().__init__(compare_against_original)
        self.allow_pos_pref = allow_pos_pref

    def get_pos(self, sent, tagset='universal'):
        '''
        :param sent: list of word strings
        tagset: {'universal', 'default'}
        :return: list of pos tags.
        Universal (Coarse) Pos tags has  12 categories
            - NOUN (nouns)
            - VERB (verbs)
            - ADJ (adjectives)
            - ADV (adverbs)
            - PRON (pronouns)
            - DET (determiners and articles)
            - ADP (prepositions and postpositions)
            - NUM (numerals)
            - CONJ (conjunctions)
            - PRT (particles)
            - . (punctuation marks)
            - X (a catch-all for other categories such as abbreviations or foreign words)
        '''

        if tagset == 'default':
            word_n_pos_list = nltk.pos_tag(sent)
        elif tagset == 'universal':
            word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply pos check constraint without `newly_modified_indices`"
            )
        for i in indices:
            reference_word = reference_text.words[i]
            # transformed_word = transformed_text.words[i]
            word_pos = self.get_pos(sent=[reference_word])[0]
            # if not (word_pos in self.allow_pos_pref and len(reference_word) > 2):
            if word_pos not in self.allow_pos_pref:
                return False

        return True

    def extra_repr_keys(self):
        return ["allow_pos_pref"] + super().extra_repr_keys()
