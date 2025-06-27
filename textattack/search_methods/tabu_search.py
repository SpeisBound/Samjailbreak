import collections
import random

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.shared import AttackedText
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from textattack.shared import utils
import numpy as np
import torch
from textattack.shared.validators import transformation_consists_of_word_swaps


class TabuSearch(SearchMethod):
    def __init__(self,
                 max_iters=50,
                 max_replacements_per_index=30,
                 window_size=12,
                 trials=1000, ):
        self.max_iters = max_iters
        self.trials = trials
        self.max_replacements_per_index = max_replacements_per_index
        # 长短修改 10 30
        self._optimization_max_iters = 50
        self._search_over = False
        self.window_size = window_size
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.use = UniversalSentenceEncoder()

        self.print = False
        self.print_simple = False

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

    def _check_position(self, current, ret_res):
        for res in ret_res:
            if len([idx for idx in current if idx not in res]) / len(current) > 0.2 and \
                    len([idx for idx in res if idx not in current]) / len(res) >= 0.4:
                continue
            else:
                return False
        return True

    def select_init_txt(self, txt_list):
        ret_res = [txt_list[0][0]]
        ret_changed_idxs = [txt_list[0][1]]
        if self.print_simple:
            print('return  all_changed list len:' + str(len(txt_list)))
            print('all idx:')
            for t in txt_list:
                print(str(t[1]))
            print('---------------------------------')
            print('return  idxs=' + str(txt_list[0][1]))
        for txt in txt_list:
            txt_changed_len = len(txt[1])
            if len(txt_list[0][1]) * 1.5 > txt_changed_len or txt_changed_len <= 4:
                if self._check_position(current=txt[1], ret_res=ret_changed_idxs):
                    ret_res.append(txt[0])
                    ret_changed_idxs.append(txt[1])
                    if self.print_simple:
                        print('return  idxs=' + str(txt[1]))
                if len(ret_res) >= 5:
                    break
        return ret_res, ret_changed_idxs

    def _generate_multi_initial_adversarial_example(self, initial_result):
        initial_text = initial_result.attacked_text
        len_text = len(initial_text.words)
        over = 0
        txt_list = []
        for trial in range(self.trials):
            random_index_order = np.arange(len_text)
            np.random.shuffle(random_index_order)
            random_text = initial_text
            random_generated_texts = []
            for idx in random_index_order:

                transformed_texts = self.get_transformations(
                    random_text, original_text=initial_text, indices_to_modify=[idx]
                )

                if len(transformed_texts) > 0:
                    random_text = np.random.choice(transformed_texts)
                    random_generated_texts.append(random_text)
            results, search_over = self.get_goal_results(random_generated_texts)
            for i in range(len(results)):
                if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    txt = self._search_space_reduction(initial_result, results[i].attacked_text)
                    txt_changed_idxs = list(initial_text.all_words_diff(txt))
                    txt_list.append((txt, txt_changed_idxs))
                    if txt_list:
                        txt_list.sort(key=lambda x: len(x[1]))
                    if len(txt_list[0][1]) <= 2 and trial >= 5:
                        over = 1
                        break
                    elif len(txt_list[0][1]) <= 5 and (trial >= 10 or len(txt_list) >= 5):
                        over = 1
                        break
                    elif len(txt_list[0][1]) <= 10 and (trial >= 15 or len(txt_list) >= 5):
                        over = 1
                        break
                    elif len(txt_list[0][1]) <= 15 and (trial >= 20 or len(txt_list) >= 10):
                        over = 1
                        break
                    elif len(txt_list[0][1]) > 15 and (trial >= 30 or len(txt_list) >= 10):
                        over = 1
                        break
            if over:
                # print('trial:' + str(trial))
                return self.select_init_txt(txt_list)
            if search_over:
                if txt_list:
                    return self.select_init_txt(txt_list)
                else:
                    return initial_text, []

        return initial_text, []

    def _search_space_reduction(self, initial_result, initial_adv_text, model='sim'):
        """Reduces the count of perturbed words in the `initial_adv_text`.

        Args:
            initial_result (GoalFunctionResult): Original result.
            initial_adv_text (int): The adversarial text obtained after random initialization.
        Returns:
            initial_adv_text as `AttackedText`
        """
        original_text = initial_result.attacked_text
        different_indices = original_text.all_words_diff(initial_adv_text)
        replacements = []
        txts = []
        # Replace the substituted words with their original counterparts.
        for idx in different_indices:

            new_text = initial_adv_text.replace_word_at_index(
                idx, initial_result.attacked_text.words[idx]
            )
            # Filter out replacements that are not adversarial.
            results, search_over = self.get_goal_results([new_text])

            if search_over:
                return initial_adv_text
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                txts.append(new_text.text)
                replacements.append((idx, initial_result.attacked_text.words[idx]))

        if len(txts) == 0:
            return initial_adv_text

        if model == 'ppl':
            pass
            # ppl_scores = self._get_ppl(txts)
            # replacements = list(zip(ppl_scores, replacements))
            # replacements.sort()
        else:
            similarity_scores = self._get_similarity_score(txts, initial_result.attacked_text.text)
            replacements = list(zip(similarity_scores, replacements))
            replacements.sort()
            replacements.reverse()

        # Sort the replacements based on semantic similarity and replace
        # the substitute words with their original counterparts till the
        # remains adversarial.

        for score, replacement in replacements:

            new_text = initial_adv_text.replace_word_at_index(
                replacement[0], replacement[1]
            )
            results, _ = self.get_goal_results([new_text])
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                initial_adv_text = new_text
            else:
                break

        return initial_adv_text

    @staticmethod
    def random_pick_idx_with_unnormalized_prob(prob):
        p = np.random.uniform(low=0.0, high=np.sum(prob))
        ret = 0
        while p >= 0 and ret < len(prob):
            p -= prob[ret]
            if p < 0:
                return ret
            ret += 1
        return len(prob) - 1

    def _get_ppl(self, atk_text):
        if not isinstance(atk_text, list):
            input_seq = torch.LongTensor(self.gpt2_tokenizer.encode(atk_text.text)).unsqueeze(0)[:, :1024]
            output = self.gpt2_model(input_seq, labels=input_seq)
            ppl = torch.exp(output.loss).tolist()
        else:
            ppl = []
            for txt in atk_text:
                input_seq = torch.LongTensor(self.gpt2_tokenizer.encode(txt)).unsqueeze(0)[:, :1024]
                output = self.gpt2_model(input_seq, labels=input_seq)
                ppl.append(torch.exp(output.loss).tolist())
        return ppl

    def _eval_func(self, transformed_texts, initial_text, transition_index):
        cos_sim = self._get_similarity_score(transformed_texts=[transformed_texts],
                                             initial_text=initial_text,
                                             window_around_idx=transition_index, use_window=True).tolist()[0]
        return cos_sim

    def _proposal(self, current_text, original_text, transition_index, pick_method='sim'):
        results_text = self.get_transformations(current_text=current_text, original_text=original_text,
                                                indices_to_modify=[transition_index])

        if len(results_text) < 1:
            return current_text
        elif len(results_text) > self.max_replacements_per_index:
            results_text = results_text[:self.max_replacements_per_index]

        if pick_method == 'sim':
            sds = []
            for result_text in results_text:
                sds.append(self._eval_func(transformed_texts=result_text, initial_text=original_text,
                                           transition_index=transition_index))
            pick_id = TabuSearch.random_pick_idx_with_unnormalized_prob(sds)
        elif pick_method == 'random':
            pick_id = random.randint(0, len(results_text) - 1)
        else:
            pick_id = 0

        return results_text[pick_id]

    def _perform_search(self, initial_result):

        initial_text = initial_result.attacked_text
        new_texts, changed_idxs = self._generate_multi_initial_adversarial_example(initial_result)
        if not changed_idxs:
            return initial_result
        init_results, search_over = self.get_goal_results(new_texts)
        if search_over:
            return initial_result
        best_res = 0
        results = None
        for k in range(0, len(init_results)):

            changed_indices = changed_idxs[k]
            new_text = init_results[k].attacked_text
            if init_results[k].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                return init_results[k]

            if self.print:
                print('初始对抗样本：')
                self.diff_color(initial_result, init_results[k])
            '''
            new_text为初始对抗样本，initial_text为源样本，changed_indices为初始相比源改变的单词个数
            '''
            changed_indices_pre_number = len(changed_indices)
            current_text = new_text
            # 快速接近，优化初始点
            if changed_indices_pre_number > 15:
                current_text, changed_indices, changed_indices_pre_number = self.fast_search(current_text, initial_text,
                                                                                             initial_result,
                                                                                             changed_indices)
            if changed_indices_pre_number > 30:
                if self.print_simple:
                    print('changed_indices_pre_number > 30')
                return initial_result
            elif changed_indices_pre_number > 15:
                results, _ = self.get_goal_results([current_text])
                print('changed_indices_pre_number > 15')
                return results[0]

            # 禁忌搜索执行
            if len(changed_indices) < 2:
                tabu_queue = collections.deque(changed_indices)
            else:
                tabu_queue = collections.deque(maxlen=int(len(changed_indices) * 0.7))
            tabu_queue_record = collections.deque()
            max_iters = 50
            best_solution_stagnant = 0
            local_opt = (current_text, [], self._get_similarity_score(transformed_texts=[new_text.text],
                                                                      initial_text=initial_text.text).tolist()[0],
                         changed_indices_pre_number)
            best_solution = local_opt
            for i in range(max_iters):
                # local opt 为[]则特赦
                if not local_opt:
                    if len(tabu_queue) != 0:
                        if len(tabu_queue_record) != 0:
                            number = tabu_queue_record.popleft()
                        else:
                            number = int(len(tabu_queue) / 2)
                        if number > 5:
                            number = int(number / 2)
                        for _ in range(number):
                            if len(tabu_queue) != 0:
                                tabu_queue.popleft()
                    best_solution_stagnant = best_solution_stagnant + 1
                else:
                    number = 0
                    for lo in local_opt[1]:
                        if lo not in tabu_queue:
                            number = number + 1
                            tabu_queue.append(lo)
                    if number != 0:
                        tabu_queue_record.append(number)
                    if local_opt[3] < best_solution[3]:
                        best_solution = local_opt
                    elif local_opt[3] == best_solution[3]:
                        if local_opt[2] > best_solution[2]:
                            best_solution = local_opt
                        else:
                            best_solution_stagnant = best_solution_stagnant + 1
                    else:
                        best_solution_stagnant = best_solution_stagnant + 1
                    current_text = local_opt[0]

                if search_over:
                    break
                changed_indices_pre_number = len(list(initial_text.all_words_diff(current_text)))
                if best_solution_stagnant == 10 or (best_solution_stagnant == 5 and changed_indices_pre_number > 15):
                    break

                if changed_indices_pre_number <= 3:
                    move_iter = 5
                # elif changed_indices_pre_number <= 6:
                #     move_iter = 2 * changed_indices_pre_number
                # elif changed_indices_pre_number <= 10:
                #     move_iter = int(2 * changed_indices_pre_number)
                else:
                    move_iter = int(2 * changed_indices_pre_number)

                free_indices = [i for i in changed_indices if i not in tabu_queue]

                if self.print:
                    print('\033[1;35mOptimal find Loop round: \033[0m' + str(i), end='   ')
                    print('\033[1;35mTabu Queue: \033[0m' + str(list(tabu_queue)))

                local_opt = []
                # 领域移动，寻找局部最优
                for move in range(move_iter):  # m轮内一直无法寻找到对抗样本/sim差距突变巨大，特赦解。如果能找到解，解为劣解中的最优解进入tabu
                    transformer_texts = []
                    succ_transformer_texts = []
                    if self.print:
                        print('\033[1;35mLocal-opt Loop round: \033[0m' + str(move))

                    for transition_index in free_indices:
                        current_text = self._proposal(current_text=current_text, original_text=initial_text,
                                                      transition_index=transition_index)
                        transformer_texts.append(current_text)

                    results, search_over = self.get_goal_results(transformer_texts)
                    for result in results:
                        if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            succ_transformer_texts.append(result.attacked_text)
                    if len(succ_transformer_texts) == 0:
                        continue
                    else:
                        opt_adv, sim_opt_adv, opt_adv_changed_indices = self.atk_text_rank(succ_transformer_texts,
                                                                                           initial_result,
                                                                                           initial_text)

                    results, search_over = self.get_goal_results([opt_adv])
                    # 1 有替换词被替换为原单词
                    if len(opt_adv_changed_indices) < changed_indices_pre_number:
                        tabu_object = [idx for idx in free_indices if idx not in opt_adv_changed_indices]
                        if self.print:
                            print('局部最优解：Sim score:' + str(sim_opt_adv) + ' tabu_object' + str(tabu_object))
                            self.diff_color(initial_result, results[0], ori_show=0, atk_show=1)
                        local_opt = (opt_adv, tabu_object, sim_opt_adv, len(opt_adv_changed_indices))
                        break


                    if not local_opt:
                        if int(len(free_indices) * 0.5) >= 4:
                            free_number = 4
                        else:
                            free_number = int(len(free_indices) * 0.5)
                        sample = random.sample(free_indices, free_number)
                        local_opt = (opt_adv, sample, sim_opt_adv, len(opt_adv_changed_indices))
                    # 2 语义有提升 改变单词个数不变
                    if sim_opt_adv > local_opt[2]:
                        if int(len(free_indices) * 0.5) >= 4:
                            free_number = 4
                        else:
                            free_number = int(len(free_indices) * 0.5)
                        sample = random.sample(free_indices, free_number)
                        local_opt = (opt_adv, sample, sim_opt_adv, len(opt_adv_changed_indices))
                        if self.print:
                            print('局部优解：Sim score:' + str(sim_opt_adv) + ' tabu_object ' + str(sample))
                            self.diff_color(initial_result, results[0], ori_show=0, atk_show=1)

                        if search_over:
                            break

            results, search_over = self.get_goal_results([best_solution[0]])
            if self.print_simple:
                print(
                    '————————————最优解查询次数：' + str(results[0].num_queries) + '  扰动词个数：' + str(
                        best_solution[3]) + '  sim: ' +
                    str(best_solution[2]))
                self.diff_color(initial_result, results[0], ori_show=1, atk_show=1)
            if best_res == 0:
                best_res = results[0]
                if self.print_simple:
                    print('当前best_result: ' + str(k))
            else:
                len1 = len(list(initial_text.all_words_diff(results[0].attacked_text)))
                len2 = len(list(initial_text.all_words_diff(best_res.attacked_text)))
                sim1 = self._get_similarity_score(transformed_texts=[results[0].attacked_text.text],
                                                  initial_text=initial_text.text).tolist()[0]
                sim2 = self._get_similarity_score(transformed_texts=[best_res.attacked_text.text],
                                                  initial_text=initial_text.text).tolist()[0]
                if len1 > len2 or (len2 == len1 and sim1 < sim2):
                    pass
                else:
                    best_res = results[0]
                    if self.print_simple:
                        print('当前best_result: ' + str(k))

        return best_res

    def fast_search(self, current_text, initial_text, initial_result, changed_indices):
        fast_search_times = 0
        changed_idxs = changed_indices
        changed_indices_pre_number = len(changed_idxs)
        best_res = current_text
        while changed_indices_pre_number > 15 and fast_search_times <= 200:
            transformer_texts = []
            succ_trans_texts = []
            if self.print:
                print('fast_search_times:' + str(fast_search_times))
            fast_search_times = fast_search_times + 1
            for transition_index in changed_idxs:
                current_text = self._proposal(current_text=current_text, original_text=initial_text,
                                              transition_index=transition_index, pick_method='random')
                transformer_texts.append(current_text)
            results, search_over = self.get_goal_results(transformer_texts)
            if search_over:
                return best_res, changed_idxs, changed_indices_pre_number
            for result in results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    succ_trans_texts.append(self._search_space_reduction(initial_result=initial_result,
                                                                         initial_adv_text=result.attacked_text,
                                                                         model='sim'))
            if len(succ_trans_texts) == 0:
                continue
            else:
                current_text, _, changed_idxs = self.atk_text_rank(succ_trans_texts, initial_result, initial_text)
                best_res = current_text
                changed_indices_pre_number = len(changed_idxs)
                if self.print:
                    print('fast-search-text-len:' + str(changed_indices_pre_number))

        if changed_indices_pre_number > 15:
            if self.print_simple:
                print('fast search failed!')
        else:
            if self.print_simple:
                print('fast search succeed!')
        return best_res, changed_idxs, changed_indices_pre_number

    def atk_text_rank(self, atk_texts, initial_result, initial_text):
        ranked_text = []
        final_idx = 0
        for text in atk_texts:
            opt_adv = self._search_space_reduction(initial_result=initial_result,
                                                   initial_adv_text=text, model='sim')
            sim_opt_adv = self._get_similarity_score(transformed_texts=[text.text],
                                                     initial_text=initial_text.text).tolist()[0]
            opt_adv_changed_indices = list(initial_text.all_words_diff(opt_adv))
            ranked_text.append((opt_adv, sim_opt_adv, opt_adv_changed_indices))
        if len(ranked_text) == 1:
            return ranked_text[0]
        ranked_text.sort(key=lambda x: len(x[2]))
        for i in range(1, len(ranked_text)):
            if ranked_text[0][2] == ranked_text[i][2]:
                if ranked_text[0][1] < ranked_text[i][1]:
                    final_idx = i
        return ranked_text[final_idx]

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
