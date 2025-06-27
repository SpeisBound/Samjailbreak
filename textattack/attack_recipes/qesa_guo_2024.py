from textattack.constraints.pre_transformation import (
    InputColumnModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.attacker import Attack
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe
from textattack.search_methods import SamplingSearch
# from textattack.search_methods.rejection_sampling import RejectionSampling


class QESAGuo2024(AttackRecipe):

    @staticmethod
    def build(model, file_logger=None, init_adv_examples=None):
        goal_function = model
        constraints = []
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Minimum word embedding cosine similarity of 0.3.
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.3))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        # constraints.append(POSCheck())

        transformation = WordSwapEmbedding(max_candidates=50)
        # if init_adv_examples:
        #     search_method = SamplingSearch(file_logger=file_logger, init_adv_examples=init_adv_examples)
        # else:
        #     search_method = SamplingSearch(file_logger=file_logger)

        search_method = SamplingSearch(file_logger=file_logger)
        return Attack(goal_function, constraints, transformation, search_method)
