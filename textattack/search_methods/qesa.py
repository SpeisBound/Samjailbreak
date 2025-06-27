import random
import numpy as np
import torch
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText
from textattack.shared import utils
from textattack.shared.validators import transformation_consists_of_word_swaps
from collections import Counter
from textattack.shared.utils import device

import criteria
from collections import defaultdict
import pickle
import os


class SamplingSearch(SearchMethod):
    def __init__(self,
                 max_replacements_per_index=30,
                 window_size=12,
                 trials=500,
                 file_logger=None,
                 init_adv_examples=None,
                 ):

        self.initialization_trials = trials
        self.max_replacements_per_index = max_replacements_per_index
        self.max_sampling_time_per_round = 30

        self.window_size = window_size
        # Similarity
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.use = UniversalSentenceEncoder()

        self.init_change_indices_number = 0
        self.initial_text = None

        self.current_index = 0
        self.index_count_list = []

        self.file_logger = file_logger
        self.llm_init_examples = False
        self.init_adv_examples = None

        self.counter_fitting_embeddings_path = "./data_temp/init/counter-fitted-vectors.txt"
        self.counter_fitting_cos_sim_path = './data_temp/init/mat.txt'
        self.top_k_words = 1000000

    def target_distribution(self, adv_texts, initial_text):
        adv_text = [adv_texts[i].text for i in range(len(adv_texts))]
        ps = 0.5
        pp = 0.5
        sims = self._get_similarity_score(transformed_texts=adv_text, initial_text=self.initial_text).tolist()[0]
        pert_number = len(initial_text.all_words_diff(adv_texts[0]))
        return ps * sims + pp * (
                    1 - (pert_number / self.init_change_indices_number))

    def _get_similarity_score(self, transformed_texts, initial_text, window_around_idx=-1, use_window=False):
        """
        Calculates semantic similarity between the `transformed_texts` and `initial_text`
        Args:
            transformed_texts (list): The perturbed or mutated text inputs.
            initial_text (AttackedText): The original text input.
            window_around_idx (int): The index around which the window to consider.
            use_window (bool): Calculate similarity with or without window.
        Returns:
            scores as `list[float]`
        """

        if use_window:
            texts_within_window = []
            for txt in transformed_texts:
                texts_within_window.append(
                    txt.text_window_around_index(window_around_idx, self.window_size)
                )
            transformed_texts = texts_within_window
            initial_text = initial_text.text_window_around_index(
                window_around_idx, self.window_size
            )
        embeddings_transformed_texts = self.use.encode(transformed_texts)
        embeddings_initial_text = self.use.encode(
            [initial_text] * len(transformed_texts)
        )

        if not isinstance(embeddings_transformed_texts, torch.Tensor):
            embeddings_transformed_texts = torch.tensor(embeddings_transformed_texts)

        if not isinstance(embeddings_initial_text, torch.Tensor):
            embeddings_initial_text = torch.tensor(embeddings_initial_text)

        scores = self.sim_metric(embeddings_transformed_texts, embeddings_initial_text)
        return scores

    # just shuffle and replace perturbed words with original words
    def _search_space_reduction(self, initial_result, adv_result):
        """Reduces the count of perturbed words in the `initial_adv_text`.

        Args:
            initial_result (GoalFunctionResult): Original result.
            initial_adv_text: The adversarial text obtained after random initialization.
        Returns:
            initial_adv_text as `AttackedText`
        """
        initial_adv_text = adv_result.attacked_text
        original_text = initial_result.attacked_text
        different_indices = list(original_text.all_words_diff(initial_adv_text))
        random.shuffle(different_indices)
        spr_results = [adv_result]
        search_over = False
        for idx in different_indices:

            new_text = initial_adv_text.replace_word_at_index(
                idx, initial_result.attacked_text.words[idx]
            )
            results, search_over = self.get_goal_results([new_text])

            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                initial_adv_text = new_text
                spr_results = results
        return initial_adv_text, spr_results, search_over

    def generate_init_adv_example(self, initial_result):
        text_ls = initial_result.attacked_text.text.split()
        embed_func = self.counter_fitting_embeddings_path
        counter_fitting_cos_sim_path = self.counter_fitting_cos_sim_path
        top_k_words = self.top_k_words
        cos_sim = None
        idx2word = {}
        word2idx = {}
        word_idx_dict = {}

        with open(embed_func, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1

        if counter_fitting_cos_sim_path:
            with open(counter_fitting_cos_sim_path, "rb") as fp:
                cos_sim = pickle.load(fp)

        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index

        embed_file = open(embed_func)
        embed_content = embed_file.readlines()

        len_text = len(text_ls)
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        for i in range(len(pos_ls)):
            if len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append([float(num) for num in embed_content[word_idx_dict[word]].strip().split()[1:]])

        synonym_words, synonym_values = [], []
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp = []
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        qrs = 0
        flag = 0
        th = 0

        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            results, search_over = self.get_goal_results([AttackedText(' '.join(random_text))])
            qrs += 1
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                flag = 1
                break
            if search_over:
                return 0, initial_result
            th += 1
            if th > len_text:
                break

        old_qrs = qrs
        while qrs < old_qrs + 20000 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            results, search_over = self.get_goal_results([AttackedText(' '.join(random_text))])
            qrs += 1
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                flag = 1
                break
            if search_over:
                return 0, initial_result

        return 1, results[0]

    def text_transformation(self, current_text, original_text, indices_to_modify):
        results_text = self.get_transformations(current_text=current_text, original_text=original_text,
                                                indices_to_modify=indices_to_modify)
        if len(results_text) < 1:
            return [current_text]
        elif len(results_text) > self.max_replacements_per_index:
            results_text = results_text[:self.max_replacements_per_index]
        return results_text

    @staticmethod
    def softmax(scores):
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=0)
        return probabilities

    def words_importance(self, idx):

        t = Counter(self.index_count_list)
        index = list(t.keys()).index(idx)
        all_times = np.array(list(t.values()))
        softmax_val = SamplingSearch.softmax(all_times)
        return softmax_val[index]

    def _perform_search(self, initial_result):
        total_sample = 0
        total_accept = 0

        get_example, init_adv_result = self.generate_init_adv_example(initial_result)
        initial_adv_text = init_adv_result.attacked_text
        if not get_example:
            print('failed to search init_adv_example!')
            return initial_result

        # Search space reduction
        new_text, results, search_over = self._search_space_reduction(initial_result, init_adv_result)

        if (initial_adv_text.num_words != initial_result.attacked_text.num_words or search_over or initial_result.
                attacked_text.num_words > 700):
            return results[0]

        initial_text = initial_result.attacked_text
        changed_indices = initial_text.all_words_diff(new_text)
        changed_indices = list(changed_indices)
        self.index_count_list = self.index_count_list + changed_indices
        self.init_change_indices_number = len(changed_indices)
        if len(changed_indices) < 1:
            return results[0]
        self.initial_text = initial_text.text

        '''
        new_text为初始对抗样本，initial_text为源样本，changed_indices为初始改变的单词个数
        '''
        sampling_times = 0
        random.shuffle(changed_indices)
        current_sample_index = 0
        next_sample_index = 1
        current_text = new_text
        current_td = self.target_distribution(adv_texts=[current_text], initial_text=initial_text)
        sim_current_text = self._get_similarity_score(transformed_texts=[current_text.text], initial_text=initial_text.text).tolist()[0]
        best_rst = (current_td, results[0])
        self.max_sampling_time_per_round = 300 * len(changed_indices)  # 10
        best_not_renew = 0
        best_not_renew_threshold = 300 * len(changed_indices)  # 10
        # print(f'all changed indices:{changed_indices}')
        search_over = False
        all_trans_rate_list = []
        accept_rate_list = []
        while sampling_times < self.max_sampling_time_per_round:
            if search_over:
                print('search over')
                break
            sampling_times += 1
            results_texts = self.text_transformation(current_text=current_text, original_text=initial_text,
                                                     indices_to_modify=[changed_indices[current_sample_index]])

            selected_index = random.randint(0, len(results_texts) - 1)
            sim_selected_text = self._get_similarity_score(transformed_texts=[results_texts[selected_index].text],
                                                           initial_text=initial_text.text).tolist()[0]
            selected_td = self.target_distribution(adv_texts=[results_texts[selected_index]], initial_text=initial_text)
            k1 = (1. - sim_selected_text)*10
            k2 = (1.-sim_current_text)*10
            pd_sm = SamplingSearch.softmax([(1.-sim_selected_text)*10, (1.-sim_current_text)*10])
            pd_transition_current2selected = pd_sm[1]
            pd_transition_selected2current = pd_sm[0]
            # print(f'pd_sm:softmax[{(1.-sim_selected_text)*10}, {(1.-sim_current_text)*10}] = [{pd_sm[1]},{pd_sm[0]}]')

            td_sm = SamplingSearch.softmax([selected_td, current_td])
            selected_td_sm = td_sm[0]
            current_td_sm = td_sm[1]
            # print(f'td_sm:softmax[{selected_td}, {current_td}] = [{td_sm[0]},{td_sm[1]}]')

            val = (selected_td_sm * pd_transition_selected2current) / (current_td_sm * pd_transition_current2selected)
            word_importance_val = self.words_importance(idx=changed_indices[current_sample_index])
            p_val = 0.8
            p_wiv = 0.2
            ratio_str = f'accept_rate:{p_val}*[{selected_td_sm}*{pd_transition_selected2current}/{current_td_sm}*{pd_transition_current2selected} = {val}] + {p_wiv}*{word_importance_val}= {(p_val * val) + (p_wiv * word_importance_val)}'
            print(ratio_str)
            val = (p_val * val) + (p_wiv * word_importance_val)

            all_trans_rate_list.append(val)
            # TODO:
            acceptance_ratio = min(1.5, val)
            total_sample += 1
            current_sample_index += 1
            next_sample_index += 1
            if current_sample_index >= len(changed_indices):
                current_sample_index = 0
            if next_sample_index >= len(changed_indices):
                next_sample_index = 0

            # print("acceptance_ratio:{:.3f}, pd_current2selected:{:.3f}, pd_selected2current:{:.3f},
            # current_td:{:.3f},selected_td:{:.3f}, last_sim_sum:{:.3f}, now_sum_sim:{:.3f}, next_sum_sim:{:.3f}".
            #       format(acceptance_ratio, pd_transition_current2selected, pd_transition_selected2current,
            #              current_td, selected_td, last_sim_sum, now_sum_sim, next_sum_sim))

            best_not_renew += 1
            # TODO:
            if np.random.uniform(0, 1.5) < acceptance_ratio:
                accept_rate_list.append(val)
                total_accept += 1
                results, search_over = self.get_goal_results([results_texts[selected_index]])
                for result in results:
                    if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        # self.diff_color(original_result=initial_result, perturbed_result=result)
                        # ssr_text, rst, search_over = self._search_space_reduction(initial_result=initial_result,
                        #                                                           adv_result=result)
                        ssr_text, rst, search_over = result.attacked_text, [result], search_over
                        current_td = self.target_distribution(adv_texts=[ssr_text], initial_text=initial_text)
                        real_td = current_td
                        current_text = ssr_text
                        sim_current_text = self._get_similarity_score(transformed_texts=[current_text.text],
                                                                      initial_text=initial_text.text).tolist()[0]
                        best_not_renew += 1
                        if real_td > best_rst[0]:
                            best_not_renew = 0
                            best_rst = (real_td, rst[0])
                            self.index_count_list = self.index_count_list + list(
                                initial_text.all_words_diff(rst[0].attacked_text))
                        sampling_times = 0
                    else:
                        current_text = results_texts[selected_index]
                        current_td = selected_td
                        sim_current_text = sim_selected_text
            if best_not_renew >= best_not_renew_threshold:
                break

        self.diff_color(original_result=initial_result, perturbed_result=best_rst[1])
        self.file_logger.insert_log(content=f'{total_sample}-{total_accept}-{best_rst[0]}\n')
        all_trans_rate_list = [f'{x:.4f}' for x in all_trans_rate_list]
        all_trans_rate_list = ' '.join(all_trans_rate_list)
        accept_rate_list = [f'{x:.4f}' for x in accept_rate_list]
        accept_rate_list = ' '.join(accept_rate_list)
        self.file_logger.insert_log(content=f'all_trans_rate_list:{all_trans_rate_list}')
        self.file_logger.insert_log(content=f'accept_rate_list:{accept_rate_list}')
        return best_rst[1]

    def diff_color(self, original_result, perturbed_result, color_method='ansi', ori_show=1, atk_show=1):
        """Highlights the difference between two texts using color.

        Has to account for deletions and insertions from original text to
        perturbed. Relies on the index map stored in
        ``self.original_result.attacked_text.attack_attrs["original_index_map"]``.
        """
        t1 = original_result.attacked_text
        t2 = perturbed_result.attacked_text

        if color_method is None:
            return t1.printable_text(), t2.printable_text()

        color_1 = original_result.get_text_color_input()
        color_2 = perturbed_result.get_text_color_perturbed()

        # iterate through and count equal/unequal words
        words_1_idxs = []
        t2_equal_idxs = set()
        original_index_map = t2.attack_attrs["original_index_map"]
        for t1_idx, t2_idx in enumerate(original_index_map):
            if t2_idx == -1:
                # add words in t1 that are not in t2
                words_1_idxs.append(t1_idx)
            else:
                w1 = t1.words[t1_idx]
                w2 = t2.words[t2_idx]
                if w1 == w2:
                    t2_equal_idxs.add(t2_idx)
                else:
                    words_1_idxs.append(t1_idx)

        # words to color in t2 are all the words that didn't have an equal,
        # mapped word in t1
        words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

        # make lists of colored words
        words_1 = [t1.words[i] for i in words_1_idxs]
        words_1 = [utils.color_text(w, color_1, color_method) for w in words_1]
        words_2 = [t2.words[i] for i in words_2_idxs]
        words_2 = [utils.color_text(w, color_2, color_method) for w in words_2]

        t1 = original_result.attacked_text.replace_words_at_indices(
            words_1_idxs, words_1
        )
        t2 = perturbed_result.attacked_text.replace_words_at_indices(
            words_2_idxs, words_2
        )

        key_color = ("bold", "underline")
        if ori_show:
            print(t1.printable_text(key_color=key_color, key_color_method=color_method))
        if atk_show:
            print(t2.printable_text(key_color=key_color, key_color_method=color_method))

    def check_transformation_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        """Returns `True` if search method does not require access to victim
        model's internal states."""
        return True

    def extra_repr_keys(self):
        return []
