import random
from collections import defaultdict

import sys
import numpy as np
import torch
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared import AttackedText
from textattack.shared.validators import transformation_consists_of_word_swaps
sys.setrecursionlimit(1000000)


class HardLabelAttackDE(PopulationBasedSearch):  # 基于种群搜索（差分进化 differential_evolution）

    def __init__(  # 初始化参数
            self,
            pop_size=30,  # 人口规模,默认为 30。
            max_iters=100,  # 要使用的最大迭代次数。 默认为 100。
            max_replacements_per_index=40,  # 索引上可以发生的最大突变数。 默认为 25。
            window_size=40,
            trials=2000,  # 生成随机对抗样本的尝试次数
            # F ,    #缩放因子[0,2]
            # CR ,   #交叉概率[0,1]
    ):
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.max_replacements_per_index = max_replacements_per_index
        self.trials = trials
        self.window_size = window_size
        self._search_over = False
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.use = UniversalSentenceEncoder()

    def _get_similarity_score(  # 定义计算语义相似度
            self, transformed_texts, initial_text, window_around_idx=-1, use_window=False
    ):
        """
        Calculates semantic similarity betweem the `transformed_texts` and `initial_text`
        计算“转换后的文本”和“初始文本”之间的语义相似度
        Args:
            transformed_texts (list): The perturbed or mutated text inputs.扰动或变异的文本输入。
            initial_text (AttackedText): The original text input.原始文本输入。
            window_around_idx (int): The index around which the window to consider.要考虑的窗口周围的索引。
            use_window (bool): Calculate similarity with or without window.计算有无窗口的相似度。
        Returns:
            scores as `list[float]`

         返回：
             分数为`list[float]`
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

    def _perturb(  # 扰动   #mutation 变异操作，也需要在_perturb函数里面实现
            self, pop_member, original_result, initial_text, index_to_perturb, best_attack
    ):
        """
        Mutates or perturbes a population member `pop_member` at `index_to_perturb` 在“扰动指数”处突变或扰动人口成员“pop_member”`
        Args:
            pop_member (PopulationMember): The pop_member to perturb or mutate.
            original_result (GoalFunctionResult): The result of `pop_member`.
            initial_text (AttackedText): The original text input.
            index_to_perturb (int): The index to be perturbed or mutated.
            best_attack (PopulationMember): The best attack until now.
        Returns:
            pop_member as `PopulationMember`
            参数：
             pop_member (PopulationMember)：要扰动或变异的 pop_member。
             original_result (GoalFunctionResult)：`pop_member` 的结果。
             initial_text (AtackedText)：原始文本输入。
             index_to_perturb (int)：要扰动或变异的索引。
             best_attack (PopulationMember)：迄今为止最好的攻击。

  返回：
             pop_member 作为 `PopulationMember`
        """
        cur_adv_text = pop_member.attacked_text

        # First try to replace the substituted word back with the original word.
        # 首先尝试将替换后的词替换回原词。
        perturbed_text = cur_adv_text.replace_word_at_index(
            index_to_perturb, initial_text.words[index_to_perturb]
        )


        results, search_over = self.get_goal_results([perturbed_text])
        if search_over:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": -1,
                    "valid_adversarial_example": False,
                },
            )
        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
            similarity_score = self._get_similarity_score(
                [perturbed_text.text], initial_text.text
            )[0]
            return PopulationMember(
                perturbed_text,
                attributes={
                    "similarity_score": similarity_score,
                    "valid_adversarial_example": True,
                },
            )
        # Replace with other synonyms.
        # 替换为其他同义词。
        perturbed_texts = self.get_transformations(
            perturbed_text,
            original_text=initial_text,
            indices_to_modify=[index_to_perturb],
        )

        final_perturbed_texts = []
        for txt in perturbed_texts:

            passed_constraints = self._check_constraints(
                txt, perturbed_text, original_text=initial_text
            )

            if (
                    passed_constraints
                    and txt.words[index_to_perturb] != cur_adv_text.words[index_to_perturb]
            ):
                final_perturbed_texts.append(txt)

        results, _ = self.get_goal_results(final_perturbed_texts)

        perturbed_adv_texts = []
        perturbed_adv_strings = []
        for i in range(len(results)):

            if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                perturbed_adv_texts.append(final_perturbed_texts[i])
                perturbed_adv_strings.append(final_perturbed_texts[i].text)

        if len(perturbed_adv_texts) == 0:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": pop_member.attributes["similarity_score"],
                    "valid_adversarial_example": False,
                },
            )

        prev_similarity_scores = self._get_similarity_score(
            [pop_member.attacked_text], initial_text, index_to_perturb, use_window=True
        )
        similarity_scores = self._get_similarity_score(
            perturbed_adv_texts, initial_text, index_to_perturb, use_window=True
        )
        best_similarity_score = -1
        for i in range(len(similarity_scores)):
            # Filter out candidates which do not improve the semantic similarity
            # score within `self.window_size`.
            # 过滤掉没有提高语义相似度的候选
            # 在`self.window_size`内打分
            if similarity_scores[i] < prev_similarity_scores[0]:
                continue

            if similarity_scores[i] > best_similarity_score:
                best_similarity_score = similarity_scores[i]
                final_adv_text = perturbed_adv_texts[i]

        if best_similarity_score == -1:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": best_similarity_score,
                    "valid_adversarial_example": False,
                },
            )

        new_pop_member = PopulationMember(
            final_adv_text,
            attributes={
                "similarity_score": best_similarity_score,
                "valid_adversarial_example": True,
            },
        )
        return new_pop_member

    def _perturbDE(self, pop_member, initial_text, index_to_perturb):
        """
        Args:
            pop_member (PopulationMember): The pop_member to perturb or mutate.
            initial_text (AttackedText): The original input text.
            index_to_perturb (int): The index to be perturbed or mutated.

        Returns:
             pop_member as `PopulationMember`

        """
        cur_adv_text = pop_member.attacked_text

        replaced_text = cur_adv_text.replace_word_at_index(
            index_to_perturb, initial_text.words[index_to_perturb]
        )
        new_replaced_text = replaced_text  # ??????????
        results, search_over = self.get_goal_results([replaced_text])

        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:

            new_words_texts = new_replaced_text
            new_words_texts = self.get_transformations(
                new_words_texts,
                original_text=initial_text,
                indices_to_modify=[index_to_perturb])
            text_results, search_over = self.get_goal_results(new_words_texts)

            #
            mutation_adv_texts = []
            mutation_adv_strings = []
            for x in range(len(text_results)):
                if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    mutation_adv_texts.append(text_results[x].attacked_text)
                    mutation_adv_strings.append(text_results[x].attacked_text.text)

            original_similarity_score = self._get_similarity_score(
                [cur_adv_text],
                initial_text,
                window_around_idx=index_to_perturb,
                use_window=True).tolist()[0]
            # use_window = True).tolist()
            score = original_similarity_score
            if len(mutation_adv_texts) >= 1:
                current_similarity_score = self._get_similarity_score(
                    mutation_adv_texts,
                    initial_text,
                    window_around_idx=index_to_perturb,
                    use_window=True).tolist()

                fusion = list(zip(current_similarity_score, mutation_adv_texts))
                # L1 = sorted(fusion, key=current_similarity_score, reverse=True)
                fusion.sort(key=lambda f: f[0], reverse=True)
                # fusion = filter(lambda x: x[0] >= get_score , fusion)
                fusion = filter(lambda f: f[0] >= score, fusion)
                fusion = (list(fusion))
                long = len(fusion)

                text_mutation = []

                if long >= 2:
                    for j in range(2):
                        text_mutation.append(fusion[j][1])
                elif long == 1:
                    for j in range(1):
                        text_mutation.append(fusion[j][1])
                # elif long == 1:
                #     for j in range(1):
                #         text_mutation.append(fusion[j][1])
                elif long == 0:
                    text_mutation.append(cur_adv_text)

                parent_similarity_score = self._get_similarity_score(
                    text_mutation,
                    initial_text,
                    window_around_idx=index_to_perturb,
                    use_window=True).tolist()

                population = []
                for s in range(len(parent_similarity_score)):
                    parent_adv_text = text_mutation[s]
                    index_to_word = parent_adv_text.words[index_to_perturb]

                    parent_pop_member = PopulationMember(
                        parent_adv_text,
                        attributes={
                            "similarity_score": parent_similarity_score[s],
                            "valid_adversarial_example": True,
                            "index": index_to_perturb,
                            "index_to_word": index_to_word
                        },
                    )
                    if parent_pop_member.attributes["valid_adversarial_example"]:
                        population.append(parent_pop_member)

                return population


    def _generate_initial_adversarial_example(self, initial_result):  # 生成初始化对抗样本
        """Generates a random initial adversarial example.
           生成一个随机的初始对抗示例。
        Args:
            initial_result (GoalFunctionResult): Original result.
        Returns:
            initial_adversarial_text as `AttackedText`
        """
        """生成一个随机的初始对抗样本。

                 参数：
                     initial_result (GoalFunctionResult)：原始结果。
                 返回：
                     initial_adversarial_text as `AttackedText`
        """
        initial_text = initial_result.attacked_text
        len_text = len(initial_text.words)
        # print( "initial_text:"+ str(initial_text.words ))
        # print(initial_result)
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
            results, _ = self.get_goal_results(random_generated_texts)

            for i in range(len(results)):
                if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return results[i].attacked_text, results[i]

        return initial_text, initial_result

    def _search_space_reduction(self, initial_result, initial_adv_text):  # 减少搜索空间
        """Reduces the count of perturbed words in the `initial_adv_text`.

        Args:
            initial_result (GoalFunctionResult): Original result.
            initial_adv_text (int): The adversarial text obtained after random initialization.
        Returns:
            initial_adv_text as `AttackedText`
        """
        """减少 `initial_adv_text` 中 干扰词的数量。

                 参数：
                     initial_result (GoalFunctionResult)：原始结果。
                     initial_adv_text (int)：随机初始化后得到的对抗文本。
                 返回：
                     initial_adv_text as `AttackedText`
        """
        original_text = initial_result.attacked_text
        different_indices = original_text.all_words_diff(initial_adv_text)

        # print("original_text:" + str(original_text), "initial_adv_text:" + str(initial_adv_text))
        replacements = []
        txts = []
        # Replace the substituted words with their original counterparts.
        # 将替换词改变为其原始对应词
        for idx in different_indices:

            new_text = initial_adv_text.replace_word_at_index(
                idx, initial_result.attacked_text.words[idx]
            )
            # print(new_text, idx,initial_result.attacked_text.words[idx])
            # Filter out replacements that are not adversarial.
            # 过滤掉不是对抗性的替代词
            results, search_over = self.get_goal_results([new_text])

            if search_over:
                return initial_adv_text
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                txts.append(new_text.text)
                replacements.append((idx, initial_result.attacked_text.words[idx]))
                # print(replacements)
            # print(initial_result.attacked_text.words[idx]) #可以替换为的原始输入的单词

        if len(txts) == 0:
            return initial_adv_text

        similarity_scores = self._get_similarity_score(
            txts, initial_result.attacked_text.text
        )

        # Sort the replacements based on semantic similarity and replace
        # the substitute words with their original counterparts till theremains adversarial.
        # 根据语义相似度对替换词进行排序，并将替换词替换为其原始对应词，直到仍然是对抗性的。
        replacements = list(zip(similarity_scores, replacements))
        replacements.sort()
        replacements.reverse()

        # print("replacements:" + str(replacements)) #打分排序

        for score, replacement in replacements:

            new_text = initial_adv_text.replace_word_at_index(
                replacement[0], replacement[1]
            )
            results, _ = self.get_goal_results([new_text])

            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                initial_adv_text = new_text
            else:
                break
        return initial_adv_text  # 减少修改词后的adv_text

    def _reduction_search(self, initial_result, best_attack):  # 减少搜索空间
        """Reduces the count of perturbed words in the `initial_adv_text`.

        Args:
            initial_result (GoalFunctionResult): Original result.
            best_attack (AttackedText): The current best adversarial example.
        Returns:
            best_attack as `AttackedText`
        """

        original_text = initial_result.attacked_text
        different_indices = original_text.all_words_diff(best_attack)

        replacements = []
        txts = []
        # Replace the substituted words with their original counterparts.
        # 将替换词改变为其原始对应词
        for idx in different_indices:

            new_text = best_attack.replace_word_at_index(
                idx, initial_result.attacked_text.words[idx]
            )
            # Filter out replacements that are not adversarial.
            # 过滤掉不是对抗性的替代词
            results, search_over = self.get_goal_results([new_text])

            if search_over:
                return best_attack
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                txts.append(new_text.text)
                replacements.append((idx, initial_result.attacked_text.words[idx]))

        if len(txts) == 0:
            return best_attack

        similarity_scores = self._get_similarity_score(
            txts, initial_result.attacked_text.text
        )

        # Sort the replacements based on semantic similarity and replace
        # the substitute words with their original counterparts till theremains adversarial.
        # 根据语义相似度对替换词进行排序，并将替换词替换为其原始对应词，直到仍然是对抗性的。
        replacements = list(zip(similarity_scores, replacements))
        replacements.sort()
        replacements.reverse()


        for score, replacement in replacements:

            new_text = best_attack.replace_word_at_index(
                replacement[0], replacement[1]
            )
            results, _ = self.get_goal_results([new_text])

            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_attack = new_text
            else:
                break
        return best_attack  # 减少修改词后的adv_text

    def _initialize_population(self, cur_adv_text, initial_text, changed_indices):  # 初始化种群
        """
        Initialize a population of size `pop_size` by mutating the indices in `changed_indices`
        通过对“已更改的”索引中的索引进行变异，初始化大小为“pop_size”的总体`
        Args:
            cur_adv_text (AttackedText): The adversarial text obtained after search space reduction step.
            initial_text (AttackedText): The original input text.
            changed_indices (list[int]): The indices changed in the `cur_adv_text` from the `initial_text`.

            cur_adv_text (AttackedText)：搜索空间缩减步骤后获得的对抗文本。  cur_adv_text = initial_adv_text #减少修改词后的adv_text,但是扰动率仍然过高
             initial_text (AtackedText)：原始输入文本。
             changed_indices (list):  `cur_adv_text`中，和`initial_text`相比，改变的索引。

        Returns:
            population as `list[PopulationMember]`
        """
        results, _ = self.get_goal_results([cur_adv_text])
        similarity_score = self._get_similarity_score(
            [cur_adv_text.text], initial_text.text
        )[0]
        initial_pop_member = PopulationMember(
            cur_adv_text,
            attributes={
                "similarity_score": similarity_score,
                "valid_adversarial_example": True,
            },
        )
        population = []
        # Randomly select `self.pop_size - 1` substituted indices for mutation.
        # 随机选择`self.pop_size - 1` 替换索引进行变异。
        indices = np.random.choice(len(changed_indices), size=self.pop_size - 1)
        for i in range(len(indices)):

            new_pop_member = self._perturb(
                initial_pop_member,
                results[0],
                initial_text,
                changed_indices[indices[i]],
                initial_pop_member,
            )

            if new_pop_member.attributes["valid_adversarial_example"]:
                population.append(new_pop_member)
        return population


    def _initialize_populationDE(self, cur_adv_text, initial_text, changed_indices):  # 初始化种群
        """
        Initialize a population of size `pop_size` by mutating the indices in `changed_indices`
        通过对“已更改的”索引中的索引进行变异，初始化大小为“pop_size”的总体`
        Args:
            cur_adv_text (AttackedText): The adversarial text obtained after search space reduction step.
            initial_text (AttackedText): The original input text.
            changed_indices (list[int]): The indices changed in the `cur_adv_text` from the `initial_text`.

            cur_adv_text (AttackedText)：搜索空间缩减步骤后获得的对抗文本。  cur_adv_text = initial_adv_text #减少修改词后的adv_text,但是扰动率仍然过高
             initial_text (AtackedText)：原始输入文本。
             changed_indices (list):  `cur_adv_text`中，和`initial_text`相比，改变的索引。

        Returns:
            population as `list[PopulationMember]`
        """
        get_index_word = []


        results, _ = self.get_goal_results([cur_adv_text])
        similarity_score = self._get_similarity_score(
            [cur_adv_text.text], initial_text.text
        )[0]
        indices = changed_indices
        get_index = ','.join(str(i) for i in indices)

        for i in indices:
            get_index_word.append(cur_adv_text.words[i])
        get_index_words = ','.join(str(txt) for txt in get_index_word)
        initial_pop_member = PopulationMember(
            cur_adv_text,
            attributes={
                "similarity_score": similarity_score,
                "valid_adversarial_example": True,
                "index": get_index,
                "index_to_word": get_index_words
            },
        )
        populations = []
        for idx in changed_indices:
            new_pop_members = self._perturbDE(
                initial_pop_member,
                initial_text,
                idx
            )
            populations.append(new_pop_members)
        asd = populations
        res = list(filter(None, asd))
        merge = sum(res, [])
        populations = merge

        return populations

    def _crossover(self, text_1, text_2, initial_text, best_attack):  # 定义 交叉操作
        """Does crossover step on `text_1` and `text_2` to yield a child text.

        Args:
            text_1 (AttackedText): Parent text 1.
            text_2 (AttackedText): Parent text 2.
            initial_text (AttackedText): The original text input.
            best_attack (PopulationMember): The best adversarial attack until now.
        Returns:
            pop_member as `PopulationMember`
        """
        """
        在`text_1`和`text_2`上进行交叉操作以产生子文本 
        best_attack (PopulationMember)：目前效果最好的对抗攻击。
        """
        index_to_replace = []
        words_to_replace = []

        for i in range(len(text_1.words)):

            if np.random.uniform() < 0.5:
                index_to_replace.append(i)
                words_to_replace.append(text_2.words[i])

        new_text = text_1.attacked_text.replace_words_at_indices(
            index_to_replace, words_to_replace
        )
        num_child_different_words = len(new_text.all_words_diff(initial_text))

        num_best_different_words = len(
            best_attack.attacked_text.all_words_diff(initial_text)
        )

        if num_child_different_words > num_best_different_words:
            return PopulationMember(
                text_1,
                attributes={
                    "similarity_score": best_attack.attributes["similarity_score"],
                    "valid_adversarial_example": False,
                },
            )
        else:
            for idx in index_to_replace:
                if new_text.words[idx] == text_1.words[idx]:
                    continue
                previous_similarity_score = self._get_similarity_score(
                    [best_attack.attacked_text],
                    initial_text,
                    window_around_idx=idx,
                    use_window=True,
                )
                new_similarity_score = self._get_similarity_score(
                    [new_text], initial_text, window_around_idx=idx, use_window=True
                )
                # filter out child texts which do not improve semantic similarity
                # 过滤掉不能提高语义相似度的子文本
                # within a `self.window_size`.
                if new_similarity_score < previous_similarity_score:
                    return PopulationMember(
                        text_1,
                        attributes={
                            "similarity_score": new_similarity_score,
                            "valid_adversarial_example": False,
                        },
                    )

            similarity_score = self._get_similarity_score(
                [new_text.text], initial_text.text
            )
            return PopulationMember(
                new_text,
                attributes={
                    "similarity_score": similarity_score,
                    "valid_adversarial_example": True,
                },
            )

    def _multi_disturbance_text_reduce_perturbation(self, initial_text, best_attack):
        """
        Args:
            initial_text (AttackedText): The original input text.
            best_attack (AttackedText): The current best adversarial example.
        Returns:
            get_final_text as `new best text`
        """
        print("wangzhe2")
        good_text_indices = initial_text.all_words_diff(best_attack)
        good_text_indices = list(good_text_indices)
        good_text_indices.sort(key=None, reverse=False)
        good_text_indices = random.sample(good_text_indices, 10)
        good_text_indices = list(good_text_indices)
        good_text_indices.sort(key=None, reverse=False)

        good_texts_test1 = []
        good_scores_test1 = []
        good_texts_test2 = []
        good_scores_test2 = []
        good_texts_test3 = []
        good_scores_test3 = []
        good_texts_test4 = []
        good_scores_test4 = []
        good_texts_test5 = []
        good_scores_test5 = []
        good_texts_test6 = []
        good_scores_test6 = []
        good_texts_test7 = []
        good_scores_test7 = []
        good_texts_test8 = []
        good_scores_test8 = []
        good_texts_test9 = []
        good_scores_test9 = []
        good_texts_test10 = []
        good_scores_test10 = []
        good_texts_test11 = []
        good_scores_test11 = []

        select_good_text1 = []
        get_select_score1 = []
        select_good_text2 = []
        get_select_score2 = []
        select_good_text3 = []
        get_select_score3 = []
        select_good_text4 = []
        get_select_score4 = []
        select_good_text5 = []
        get_select_score5 = []
        select_good_text6 = []
        get_select_score6 = []
        select_good_text7 = []
        get_select_score7 = []
        select_good_text8 = []
        get_select_score8 = []
        select_good_text9 = []
        get_select_score9 = []
        select_good_text10 = []
        get_select_score10 = []
        select_good_text11 = []
        get_select_score11 = []

        for i in good_text_indices:  # 替换一个
            test_text1 = best_attack.replace_words_at_indices(
                [i], [initial_text.words[i]]
            )
            test_text_result1, search_over = self.get_goal_results([test_text1])
            if test_text_result1[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                print("wangzhe4")
                good_texts_test1.append(test_text_result1[0].attacked_text)

        if len(good_texts_test1) >= 1:
            print("1")
            print("len(good_texts_test1)", len(good_texts_test1))
            for txt in good_texts_test1:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text2 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result2, search_over = self.get_goal_results([test_text2])
                    if test_text_result2[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text2.append(test_text_result2[0].attacked_text)
                        # good_texts_test2.append(test_text_result2[0].attacked_text)
                if len(select_good_text2) >= 1:
                    for txt2 in select_good_text2:
                        similarity_score = self._get_similarity_score(
                            [txt2.text], initial_text.text
                        ).tolist()[0]
                        get_select_score2.append(similarity_score)
                    final_information = list(zip(get_select_score2, select_good_text2))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test2.append(get_final_text)

        else:
            return best_attack
        if len(good_texts_test2) >= 1:
            print("2")
            for txt in good_texts_test2:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                if len(test_indexs) >= 10:
                    test_indexs = random.sample(test_indexs, 9)
                    test_indexs = list(test_indexs)
                else:
                    test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text3 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result3, search_over = self.get_goal_results([test_text3])
                    if test_text_result3[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text3.append(test_text_result3[0].attacked_text)
                        # good_texts_test3.append(test_text_result3[0].attacked_text)
                if len(select_good_text3) >= 1:
                    for txt3 in select_good_text3:
                        similarity_score = self._get_similarity_score(
                            [txt3.text], initial_text.text
                        ).tolist()[0]
                        get_select_score3.append(similarity_score)
                    final_information = list(zip(get_select_score3, select_good_text3))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test3.append(get_final_text)


        else:
            for txt in good_texts_test1:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test1.append(similarity_score)
                final_information = list(zip(good_scores_test1, good_texts_test1))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test3) >= 1:
            print("3")
            print("good_texts_test3", len(good_texts_test3))
            for txt in good_texts_test3:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                if len(test_indexs) >= 10:
                    test_indexs = random.sample(test_indexs, 9)
                    test_indexs = list(test_indexs)
                else:
                    test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text4 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result4, search_over = self.get_goal_results([test_text4])
                    if test_text_result4[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text4.append(test_text_result3[0].attacked_text)
                if len(select_good_text4) >= 1:
                    for txt4 in select_good_text4:
                        similarity_score = self._get_similarity_score(
                            [txt4.text], initial_text.text
                        ).tolist()[0]
                        get_select_score4.append(similarity_score)
                    final_information = list(zip(get_select_score4, select_good_text4))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test4.append(get_final_text)

        else:
            for txt in good_texts_test2:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test2.append(similarity_score)
                final_information = list(zip(good_scores_test2, good_texts_test2))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test4) >= 1:
            print("4")
            print("good_texts_test4", len(good_texts_test4))
            for txt in good_texts_test4:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                if len(test_indexs) >= 10:
                    test_indexs = random.sample(test_indexs, 9)
                    test_indexs = list(test_indexs)
                else:
                    test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text5 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result5, search_over = self.get_goal_results([test_text5])
                    if test_text_result5[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text5.append(test_text_result5[0].attacked_text)
                if len(select_good_text5) >= 1:
                    for txt5 in select_good_text5:
                        similarity_score = self._get_similarity_score(
                            [txt5.text], initial_text.text
                        ).tolist()[0]
                        get_select_score5.append(similarity_score)
                    final_information = list(zip(get_select_score5, select_good_text5))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test5.append(get_final_text)
        else:
            for txt in good_texts_test3:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test3.append(similarity_score)
                final_information = list(zip(good_scores_test3, good_texts_test3))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test5) >= 1:
            print("5")
            print("good_texts_test5", len(good_texts_test5))
            for txt in good_texts_test5:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                if len(test_indexs) >= 10:
                    test_indexs = random.sample(test_indexs, 9)
                    test_indexs = list(test_indexs)
                else:
                    test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text6 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result6, search_over = self.get_goal_results([test_text6])
                    if test_text_result6[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text6.append(test_text_result6[0].attacked_text)
                if len(select_good_text6) >= 1:
                    for txt6 in select_good_text6:
                        similarity_score = self._get_similarity_score(
                            [txt6.text], initial_text.text
                        ).tolist()[0]
                        get_select_score6.append(similarity_score)
                    final_information = list(zip(get_select_score6, select_good_text6))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test6.append(get_final_text)
        else:
            for txt in good_texts_test4:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test4.append(similarity_score)
                final_information = list(zip(good_scores_test4, good_texts_test4))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test6) >= 1:
            print("6")
            print("len(good_texts_test6)", len(good_texts_test6))
            for txt in good_texts_test6:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text7 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result7, search_over = self.get_goal_results([test_text7])
                    if test_text_result7[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text7.append(test_text_result7[0].attacked_text)
                if len(select_good_text7) >= 1:
                    for txt7 in select_good_text7:
                        similarity_score = self._get_similarity_score(
                            [txt7.text], initial_text.text
                        ).tolist()[0]
                        get_select_score7.append(similarity_score)
                    final_information = list(zip(get_select_score7, select_good_text7))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test7.append(get_final_text)
        else:
            for txt in good_texts_test5:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test5.append(similarity_score)
                final_information = list(zip(good_scores_test5, good_texts_test5))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test7) >= 1:
            print("7")
            print("len(good_texts_test7)", len(good_texts_test7))
            for txt in good_texts_test7:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text8 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result8, search_over = self.get_goal_results([test_text8])
                    if test_text_result8[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text8.append(test_text_result8[0].attacked_text)
                if len(select_good_text8) >= 1:
                    for txt8 in select_good_text8:
                        similarity_score = self._get_similarity_score(
                            [txt8.text], initial_text.text
                        ).tolist()[0]
                        get_select_score8.append(similarity_score)
                    final_information = list(zip(get_select_score8, select_good_text8))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test8.append(get_final_text)

        else:
            for txt in good_texts_test6:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test6.append(similarity_score)
                final_information = list(zip(good_scores_test6, good_texts_test6))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test8) >= 1:
            print("8")
            print("len(good_texts_test8)", len(good_texts_test8))
            for txt in good_texts_test8:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text9 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result9, search_over = self.get_goal_results([test_text9])
                    if test_text_result9[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text9.append(test_text_result9[0].attacked_text)
                if len(select_good_text9) >= 1:
                    for txt9 in select_good_text9:
                        similarity_score = self._get_similarity_score(
                            [txt9.text], initial_text.text
                        ).tolist()[0]
                        get_select_score9.append(similarity_score)
                    final_information = list(zip(get_select_score9, select_good_text9))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test9.append(get_final_text)

        else:
            for txt in good_texts_test7:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test7.append(similarity_score)
                final_information = list(zip(good_scores_test7, good_texts_test7))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test9) >= 1:
            print("9")
            print("len(good_texts_test9)", len(good_texts_test9))
            for txt in good_texts_test9:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text10 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result10, search_over = self.get_goal_results([test_text10])
                    if test_text_result10[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text10.append(test_text_result10[0].attacked_text)
                if len(select_good_text10) >= 1:
                    for txt10 in select_good_text10:
                        similarity_score = self._get_similarity_score(
                            [txt10.text], initial_text.text
                        ).tolist()[0]
                        get_select_score10.append(similarity_score)
                    final_information = list(zip(get_select_score10, select_good_text10))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test10.append(get_final_text)

        else:
            for txt in good_texts_test8:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test8.append(similarity_score)
                final_information = list(zip(good_scores_test8, good_texts_test8))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test10) >= 1:
            print("10")
            print("len(good_texts_test10)", len(good_texts_test10))
            for txt in good_texts_test10:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text11 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result11, search_over = self.get_goal_results([test_text11])
                    if test_text_result11[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text11.append(test_text_result11[0].attacked_text)
                if len(select_good_text11) >= 1:
                    for txt11 in select_good_text11:
                        similarity_score = self._get_similarity_score(
                            [txt11.text], initial_text.text
                        ).tolist()[0]
                        get_select_score11.append(similarity_score)
                    final_information = list(zip(get_select_score11, select_good_text11))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test11.append(get_final_text)

        else:
            for txt in good_texts_test9:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test9.append(similarity_score)
                final_information = list(zip(good_scores_test9, good_texts_test9))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test11) >= 1:
            print("11")
            print("len(good_texts_test11)", len(good_texts_test11))
            for txt in good_texts_test11:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test11.append(similarity_score)
                final_information = list(zip(good_scores_test11, good_texts_test11))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        else:
            for txt in good_texts_test10:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test10.append(similarity_score)
                final_information = list(zip(good_scores_test10, good_texts_test10))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

    def _reduce_perturbation(self, initial_text, best_attack, ):
        """
        Args:
            initial_text (AttackedText): The original input text.
            best_attack (AttackedText): The current best adversarial example.
        Returns:
            get_final_text as `AttackedText`
        """
        good_text_indices = initial_text.all_words_diff(best_attack)
        good_text_indices = list(good_text_indices)
        print("(good_text_indices)",len(good_text_indices))
        good_text_indices.sort(key=None, reverse=False)
        good_texts_test1 = []
        good_scores_test1 = []
        good_texts_test2 = []
        good_scores_test2 = []
        good_texts_test3 = []
        good_scores_test3 = []
        good_texts_test4 = []
        good_scores_test4 = []
        good_texts_test5 = []
        good_scores_test5 = []
        good_texts_test6 = []
        good_scores_test6 = []
        good_texts_test7 = []
        good_scores_test7 = []
        good_texts_test8 = []
        good_scores_test8 = []
        good_texts_test9 = []
        good_scores_test9 = []
        good_texts_test10 = []
        good_scores_test10 = []
        good_texts_test11 = []
        good_scores_test11 = []
        good_texts_test12 = []
        good_scores_test12 = []
        good_texts_test13 = []
        good_scores_test13 = []
        good_texts_test14 = []
        good_scores_test14 = []
        good_texts_test15 = []
        good_scores_test15 = []
        good_texts_test16 = []
        good_scores_test16 = []
        good_texts_test17 = []
        good_scores_test17 = []
        good_texts_test18 = []
        good_scores_test18 = []
        good_texts_test19 = []
        good_scores_test19 = []
        good_texts_test20 = []
        good_scores_test20 = []
        good_texts_test21 = []
        good_scores_test21 = []

        select_good_text1 = []
        get_select_score1 = []
        select_good_text2 = []
        get_select_score2 = []
        select_good_text3 = []
        get_select_score3 = []
        select_good_text4 = []
        get_select_score4 = []
        select_good_text5 = []
        get_select_score5 = []
        select_good_text6 = []
        get_select_score6 = []
        select_good_text7 = []
        get_select_score7 = []
        select_good_text8 = []
        get_select_score8 = []
        select_good_text9 = []
        get_select_score9 = []
        select_good_text10 = []
        get_select_score10 = []
        select_good_text11 = []
        get_select_score11 = []
        select_good_text12 = []
        get_select_score12 = []
        select_good_text13 = []
        get_select_score13 = []
        select_good_text14 = []
        get_select_score14 = []
        select_good_text15 = []
        get_select_score15 = []
        select_good_text16 = []
        get_select_score16 = []
        select_good_text17 = []
        get_select_score17 = []
        select_good_text18 = []
        get_select_score18 = []
        select_good_text19 = []
        get_select_score19 = []
        select_good_text20 = []
        get_select_score20 = []
        select_good_text21 = []
        get_select_score21 = []

        for i in good_text_indices:  # 替换一个
            test_text1 = best_attack.replace_words_at_indices(
                [i], [initial_text.words[i]]
            )
            test_text_result1, search_over = self.get_goal_results([test_text1])
            if test_text_result1[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                good_texts_test1.append(test_text_result1[0].attacked_text)

        if len(good_texts_test1) >= 1:
            print("1")
            print("len(good_texts_test1)", len(good_texts_test1))
            for txt in good_texts_test1:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text2 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result2, search_over = self.get_goal_results([test_text2])
                    if test_text_result2[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text2.append(test_text_result2[0].attacked_text)
                        # good_texts_test2.append(test_text_result2[0].attacked_text)
                if len(select_good_text2) >= 1:
                    for txt2 in select_good_text2:
                        similarity_score = self._get_similarity_score(
                            [txt2.text], initial_text.text
                        ).tolist()[0]
                        get_select_score2.append(similarity_score)
                    final_information = list(zip(get_select_score2, select_good_text2))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test2.append(get_final_text)

        else:
            print("return best_attack")
            return best_attack
        if len(good_texts_test2) >= 1:
            print("2")
            print("len(good_texts_test2)", len(good_texts_test2))
            for txt in good_texts_test2:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text3 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result3, search_over = self.get_goal_results([test_text3])
                    if test_text_result3[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text3.append(test_text_result3[0].attacked_text)
                    # good_texts_test3.append(test_text_result3[0].attacked_text)
                if len(select_good_text3) >= 1:
                    for txt3 in select_good_text3:
                        similarity_score = self._get_similarity_score(
                            [txt3.text], initial_text.text
                        ).tolist()[0]
                        get_select_score3.append(similarity_score)
                    final_information = list(zip(get_select_score3, select_good_text3))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test3.append(get_final_text)


        else:
            for txt in good_texts_test1:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test1.append(similarity_score)
                final_information = list(zip(good_scores_test1, good_texts_test1))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test3) >= 1:
            print("3")
            print("good_texts_test3", len(good_texts_test3))
            for txt in good_texts_test3:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text4 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result4, search_over = self.get_goal_results([test_text4])
                    if test_text_result4[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text4.append(test_text_result4[0].attacked_text)
                        # good_texts_test4.append(test_text_result4[0].attacked_text)
                if len(select_good_text4) >= 1:
                    for txt4 in select_good_text4:
                        similarity_score = self._get_similarity_score(
                            [txt4.text], initial_text.text
                        ).tolist()[0]
                        get_select_score4.append(similarity_score)
                    final_information = list(zip(get_select_score4, select_good_text4))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test4.append(get_final_text)

        else:
            for txt in good_texts_test2:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test2.append(similarity_score)
                final_information = list(zip(good_scores_test2, good_texts_test2))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test4) >= 1:
            print("4")
            print("len(good_texts_test4)", len(good_texts_test4))
            for txt in good_texts_test4:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text5 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result5, search_over = self.get_goal_results([test_text5])
                    if test_text_result5[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text5.append(test_text_result5[0].attacked_text)
                if len(select_good_text5) >= 1:
                    for txt5 in select_good_text5:
                        similarity_score = self._get_similarity_score(
                            [txt5.text], initial_text.text
                        ).tolist()[0]
                        get_select_score5.append(similarity_score)
                    final_information = list(zip(get_select_score5, select_good_text5))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test5.append(get_final_text)
        else:
            for txt in good_texts_test3:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test3.append(similarity_score)
                final_information = list(zip(good_scores_test3, good_texts_test3))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test5) >= 1:
            print("5")
            print("len(good_texts_test5)", len(good_texts_test5))
            for txt in good_texts_test5:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text6 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result6, search_over = self.get_goal_results([test_text6])
                    if test_text_result6[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text6.append(test_text_result6[0].attacked_text)
                if len(select_good_text6) >= 1:
                    for txt6 in select_good_text6:
                        similarity_score = self._get_similarity_score(
                            [txt6.text], initial_text.text
                        ).tolist()[0]
                        get_select_score6.append(similarity_score)
                    final_information = list(zip(get_select_score6, select_good_text6))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test6.append(get_final_text)

        else:
            for txt in good_texts_test4:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test4.append(similarity_score)
                final_information = list(zip(good_scores_test4, good_texts_test4))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test6) >= 1:
            print("6")
            print("len(good_texts_test6)", len(good_texts_test6))
            for txt in good_texts_test6:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text7 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result7, search_over = self.get_goal_results([test_text7])
                    if test_text_result7[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text7.append(test_text_result7[0].attacked_text)
                if len(select_good_text7) >= 1:
                    for txt7 in select_good_text7:
                        similarity_score = self._get_similarity_score(
                            [txt7.text], initial_text.text
                        ).tolist()[0]
                        get_select_score7.append(similarity_score)
                    final_information = list(zip(get_select_score7, select_good_text7))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test7.append(get_final_text)
        else:
            for txt in good_texts_test5:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test5.append(similarity_score)
                final_information = list(zip(good_scores_test5, good_texts_test5))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test7) >= 1:
            print("7")
            print("len(good_texts_test7)", len(good_texts_test7))
            for txt in good_texts_test7:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text8 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result8, search_over = self.get_goal_results([test_text8])
                    if test_text_result8[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text8.append(test_text_result8[0].attacked_text)
                if len(select_good_text8) >= 1:
                    for txt8 in select_good_text8:
                        similarity_score = self._get_similarity_score(
                            [txt8.text], initial_text.text
                        ).tolist()[0]
                        get_select_score8.append(similarity_score)
                    final_information = list(zip(get_select_score8, select_good_text8))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test8.append(get_final_text)

        else:
            for txt in good_texts_test6:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test6.append(similarity_score)
                final_information = list(zip(good_scores_test6, good_texts_test6))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test8) >= 1:
            print("8")
            print("len(good_texts_test8)", len(good_texts_test8))
            for txt in good_texts_test8:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text9 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result9, search_over = self.get_goal_results([test_text9])
                    if test_text_result9[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text9.append(test_text_result9[0].attacked_text)
                if len(select_good_text9) >= 1:
                    for txt9 in select_good_text9:
                        similarity_score = self._get_similarity_score(
                            [txt9.text], initial_text.text
                        ).tolist()[0]
                        get_select_score9.append(similarity_score)
                    final_information = list(zip(get_select_score9, select_good_text9))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test9.append(get_final_text)

        else:
            for txt in good_texts_test7:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test7.append(similarity_score)
                final_information = list(zip(good_scores_test7, good_texts_test7))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test9) >= 1:
            print("9")
            print("len(good_texts_test9)", len(good_texts_test9))
            for txt in good_texts_test9:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text10 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result10, search_over = self.get_goal_results([test_text10])
                    if test_text_result10[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text10.append(test_text_result10[0].attacked_text)
                if len(select_good_text10) >= 1:
                    for txt10 in select_good_text10:
                        similarity_score = self._get_similarity_score(
                            [txt10.text], initial_text.text
                        ).tolist()[0]
                        get_select_score10.append(similarity_score)
                    final_information = list(zip(get_select_score10, select_good_text10))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test10.append(get_final_text)

        else:
            for txt in good_texts_test8:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test8.append(similarity_score)
                final_information = list(zip(good_scores_test8, good_texts_test8))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test10) >= 1:
            print("10")
            print("len(good_texts_test10)", len(good_texts_test10))
            for txt in good_texts_test10:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text11 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result11, search_over = self.get_goal_results([test_text11])
                    if test_text_result11[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text11.append(test_text_result11[0].attacked_text)
                if len(select_good_text11) >= 1:
                    for txt11 in select_good_text11:
                        similarity_score = self._get_similarity_score(
                            [txt11.text], initial_text.text
                        ).tolist()[0]
                        get_select_score11.append(similarity_score)
                    final_information = list(zip(get_select_score11, select_good_text11))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test11.append(get_final_text)
        else:
            for txt in good_texts_test9:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test9.append(similarity_score)
                final_information = list(zip(good_scores_test9, good_texts_test9))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test11) >= 1:
            print("11")
            print("len(good_texts_test11)", len(good_texts_test11))
            for txt in good_texts_test11:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text12 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result12, search_over = self.get_goal_results([test_text12])
                    if test_text_result12[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text12.append(test_text_result12[0].attacked_text)
                if len(select_good_text12) >= 1:
                    for txt12 in select_good_text12:
                        similarity_score = self._get_similarity_score(
                            [txt12.text], initial_text.text
                        ).tolist()[0]
                        get_select_score12.append(similarity_score)
                    final_information = list(zip(get_select_score12, select_good_text12))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test12.append(get_final_text)
        else:
            for txt in good_texts_test10:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test10.append(similarity_score)
                final_information = list(zip(good_scores_test10, good_texts_test10))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test12) >= 1:
            print("12")
            print("len(good_texts_test12)", len(good_texts_test12))
            for txt in good_texts_test12:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text13 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result13, search_over = self.get_goal_results([test_text13])
                    if test_text_result13[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text13.append(test_text_result13[0].attacked_text)
                if len(select_good_text13) >= 1:
                    for txt13 in select_good_text13:
                        similarity_score = self._get_similarity_score(
                            [txt13.text], initial_text.text
                        ).tolist()[0]
                        get_select_score13.append(similarity_score)
                    final_information = list(zip(get_select_score13, select_good_text13))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test13.append(get_final_text)
        else:
            for txt in good_texts_test11:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test11.append(similarity_score)
                final_information = list(zip(good_scores_test11, good_texts_test11))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test13) >= 1:
            print("13")
            print("len(good_texts_test13)", len(good_texts_test13))
            for txt in good_texts_test13:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text14 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result14, search_over = self.get_goal_results([test_text14])
                    if test_text_result14[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text14.append(test_text_result14[0].attacked_text)
                if len(select_good_text14) >= 1:
                    for txt14 in select_good_text14:
                        similarity_score = self._get_similarity_score(
                            [txt14.text], initial_text.text
                        ).tolist()[0]
                        get_select_score14.append(similarity_score)
                    final_information = list(zip(get_select_score14, select_good_text14))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test14.append(get_final_text)
        else:
            for txt in good_texts_test12:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test12.append(similarity_score)
                final_information = list(zip(good_scores_test12, good_texts_test12))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test14) >= 1:
            print("14")
            print("len(good_texts_test14)", len(good_texts_test14))
            for txt in good_texts_test14:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text15 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result15, search_over = self.get_goal_results([test_text15])
                    if test_text_result15[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text15.append(test_text_result15[0].attacked_text)
                if len(select_good_text15) >= 1:
                    for txt15 in select_good_text15:
                        similarity_score = self._get_similarity_score(
                            [txt15.text], initial_text.text
                        ).tolist()[0]
                        get_select_score15.append(similarity_score)
                    final_information = list(zip(get_select_score15, select_good_text15))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test15.append(get_final_text)
        else:
            for txt in good_texts_test13:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test13.append(similarity_score)
                final_information = list(zip(good_scores_test13, good_texts_test13))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test15) >= 1:
            print("15")
            print("len(good_texts_test15)", len(good_texts_test15))
            for txt in good_texts_test15:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text16 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result16, search_over = self.get_goal_results([test_text16])
                    if test_text_result16[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text16.append(test_text_result16[0].attacked_text)
                if len(select_good_text16) >= 1:
                    for txt16 in select_good_text16:
                        similarity_score = self._get_similarity_score(
                            [txt16.text], initial_text.text
                        ).tolist()[0]
                        get_select_score16.append(similarity_score)
                    final_information = list(zip(get_select_score16, select_good_text16))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test16.append(get_final_text)
        else:
            for txt in good_texts_test14:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test14.append(similarity_score)
                final_information = list(zip(good_scores_test14, good_texts_test14))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test16) >= 1:
            print("16")
            print("len(good_texts_test16)", len(good_texts_test16))
            for txt in good_texts_test16:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text17 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result17, search_over = self.get_goal_results([test_text17])
                    if test_text_result17[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text17.append(test_text_result17[0].attacked_text)
                if len(select_good_text17) >= 1:
                    for txt17 in select_good_text17:
                        similarity_score = self._get_similarity_score(
                            [txt17.text], initial_text.text
                        ).tolist()[0]
                        get_select_score17.append(similarity_score)
                    final_information = list(zip(get_select_score17, select_good_text17))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test17.append(get_final_text)
        else:
            for txt in good_texts_test15:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test15.append(similarity_score)
                final_information = list(zip(good_scores_test15, good_texts_test15))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test17) >= 1:
            print("17")
            print("len(good_texts_test17)", len(good_texts_test17))
            for txt in good_texts_test17:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text18 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result18, search_over = self.get_goal_results([test_text18])
                    if test_text_result18[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text18.append(test_text_result18[0].attacked_text)
                if len(select_good_text18) >= 1:
                    for txt18 in select_good_text18:
                        similarity_score = self._get_similarity_score(
                            [txt18.text], initial_text.text
                        ).tolist()[0]
                        get_select_score18.append(similarity_score)
                    final_information = list(zip(get_select_score18, select_good_text18))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test18.append(get_final_text)
        else:
            for txt in good_texts_test16:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test16.append(similarity_score)
                final_information = list(zip(good_scores_test16, good_texts_test16))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test18) >= 1:
            print("18")
            print("len(good_texts_test18)", len(good_texts_test18))
            for txt in good_texts_test18:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text19 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result19, search_over = self.get_goal_results([test_text19])
                    if test_text_result19[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text19.append(test_text_result19[0].attacked_text)
                if len(select_good_text19) >= 1:
                    for txt19 in select_good_text19:
                        similarity_score = self._get_similarity_score(
                            [txt19.text], initial_text.text
                        ).tolist()[0]
                        get_select_score19.append(similarity_score)
                    final_information = list(zip(get_select_score19, select_good_text19))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test19.append(get_final_text)
        else:
            for txt in good_texts_test17:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test17.append(similarity_score)
                final_information = list(zip(good_scores_test17, good_texts_test17))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        if len(good_texts_test19) >= 1:
            print("19")
            print("len(good_texts_test19)", len(good_texts_test19))
            for txt in good_texts_test19:
                test_indexs = initial_text.all_words_diff(txt)
                test_indexs = list(test_indexs)
                for i in test_indexs:
                    test_text20 = txt.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result20, search_over = self.get_goal_results([test_text20])
                    if test_text_result20[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_good_text20.append(test_text_result20[0].attacked_text)
                if len(select_good_text20) >= 1:
                    for txt20 in select_good_text20:
                        similarity_score = self._get_similarity_score(
                            [txt20.text], initial_text.text
                        ).tolist()[0]
                        get_select_score20.append(similarity_score)
                    final_information = list(zip(get_select_score20, select_good_text20))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    good_texts_test20.append(get_final_text)
        else:
            for txt in good_texts_test18:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test18.append(similarity_score)
                final_information = list(zip(good_scores_test18, good_texts_test18))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text
        if len(good_texts_test20) >= 1:
            print("20")
            print("len(good_texts_test20)", len(good_texts_test20))
            for txt in good_texts_test20:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test20.append(similarity_score)
                final_information = list(zip(good_scores_test20, good_texts_test20))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

        else:
            for txt in good_texts_test19:
                similarity_score = self._get_similarity_score(
                    [txt.text], initial_text.text
                ).tolist()[0]
                good_scores_test19.append(similarity_score)
                final_information = list(zip(good_scores_test19, good_texts_test19))
                final_information.sort(key=lambda score: score[0], reverse=True)
                get_final_text = final_information[0][1]
                return get_final_text

    def _direct_optimization(self, initial_text, initial_result, best_attack):
        """
        Args:
            initial_text (AttackedText): The original input text.
            best_attack (AttackedText): The current best adversarial example.
        Returns:
            result as `AttackedText`
        """
        new_text = self._reduction_search(initial_result, best_attack)
        results, search_over = self.get_goal_results([new_text])
        text_indices = initial_text.all_words_diff(results[0].attacked_text)
        text_indices = list(text_indices)
        text_indices.sort(key=None, reverse=False)
        long_text_indices = len(text_indices)
        if 1 <= long_text_indices <= 6:
            print("1 <= long_text_indices <= 10")
            get_good_text = self._reduce_perturbation(initial_text, results[0].attacked_text)
            get_good_text_result, search_over = self.get_goal_results([get_good_text])
            return get_good_text_result[0]  # 直接组合优化
        if long_text_indices >= 7:  # 降低扰动率
            print("long_text_indices >= 11")
            get_good_text = self._reduction_search(initial_result, results[0].attacked_text)
            get_good_text_results0, search_over = self.get_goal_results([get_good_text])
            if search_over:
                return results[0]
            if get_good_text_results0[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                text_indices = initial_text.all_words_diff(get_good_text)
                text_indices = list(text_indices)
                text_indices.sort(key=None, reverse=False)
                long_text_indices = len(text_indices)
                if long_text_indices <= 6:
                    print("long_text_indices <= 10:", 2)
                    get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                    get_best_text_result, search_over = self.get_goal_results([get_best_text])
                    return get_best_text_result[0]
                if long_text_indices >= 7:
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text_results1, search_over = self.get_goal_results([get_good_text])
                    if search_over:
                        return get_good_text_results0[0]
                    if get_good_text_results1[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        text_indices = initial_text.all_words_diff(get_good_text)
                        text_indices = list(text_indices)
                        text_indices.sort(key=None, reverse=False)
                        long_text_indices = len(text_indices)
                        if long_text_indices <= 10:
                            print("long_text_indices <= 10:", 3)
                            get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                            get_best_text_result, search_over = self.get_goal_results([get_best_text])
                            return get_best_text_result[0]
                        if long_text_indices >= 11:
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text_results2, search_over = self.get_goal_results([get_good_text])
                            if search_over:
                                return get_good_text_results1[0]
                            if get_good_text_results2[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                text_indices = initial_text.all_words_diff(get_good_text)
                                text_indices = list(text_indices)
                                text_indices.sort(key=None, reverse=False)
                                long_text_indices = len(text_indices)
                                if long_text_indices <= 9:
                                    print("long_text_indices <= 10:", 4)
                                    get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                                    get_best_text_result, search_over = self.get_goal_results([get_best_text])
                                    return get_best_text_result[0]
                                if long_text_indices >= 10:
                                    get_good_text = self._reduction_search(initial_result, get_good_text)
                                    get_good_text_results3, search_over = self.get_goal_results([get_good_text])
                                    if search_over:
                                        return get_good_text_results2[0]
                                    if get_good_text_results3[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                        text_indices = initial_text.all_words_diff(get_good_text)
                                        text_indices = list(text_indices)
                                        text_indices.sort(key=None, reverse=False)
                                        long_text_indices = len(text_indices)
                                        if long_text_indices <= 9:
                                            print("long_text_indices <= 10:", 5)
                                            get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                                            get_best_text_result4, search_over = self.get_goal_results(
                                                [get_best_text])
                                            return get_best_text_result4[0]
                                        if long_text_indices >= 10:
                                            get_good_text = self._reduction_search(initial_result,
                                                                                   get_good_text)
                                            get_good_text_results4, search_over = self.get_goal_results(
                                                [get_good_text])
                                            if search_over:
                                                return get_good_text_results3[0]
                                            if get_good_text_results4[
                                                0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                text_indices = initial_text.all_words_diff(get_good_text)
                                                text_indices = list(text_indices)
                                                text_indices.sort(key=None, reverse=False)
                                                long_text_indices = len(text_indices)
                                                if long_text_indices <= 8:
                                                    print("long_text_indices <= 11:", 6)
                                                    get_best_text = self._reduce_perturbation(
                                                        initial_text,
                                                        get_good_text)
                                                    get_best_text_result7, search_over = self.get_goal_results(
                                                        [get_best_text])
                                                    return get_best_text_result7[0]
                                                else:
                                                    print("wangzhe1")
                                                    get_best_text = self._multi_disturbance_text_reduce_perturbation(
                                                        initial_text,
                                                        get_good_text_results4[0].attacked_text)
                                                    # get_best_text = self._reduce_perturbation(initial_text,get_good_text_results6[0].attacked_text)
                                                    get_best_text_result8, search_over = self.get_goal_results(
                                                        [get_best_text])
                                                    return get_best_text_result8[0]

    def _level_0_optimization(self, initial_text, initial_result, best_attack):
        """
        Args:
            initial_text (AttackedText): The original input text.
            initial_result (GoalFunctionResult): Original result.
            best_attack (AttackedText): The current best adversarial example.
        Returns:
            result as `AttackedText`
        """
        new_text = self._reduction_search(initial_result, best_attack)
        results, search_over = self.get_goal_results([new_text])
        text_indices = initial_text.all_words_diff(results[0].attacked_text)
        text_indices = list(text_indices)
        text_indices.sort(key=None, reverse=False)
        long_text_indices = len(text_indices)

        if 1 <= long_text_indices <= 6:
            print("1 <= long_text_indices <= 6")
            get_good_text = self._reduce_perturbation(initial_text, results[0].attacked_text)
            get_good_text_result, search_over = self.get_goal_results([get_good_text])
            return get_good_text_result[0]  # 直接组合优化

        if 7 <= long_text_indices <= 11:
            print("7 <= long_good_text_indices <= 11")
            get_good_text = self._reduction_search(initial_result, results[0].attacked_text)
            get_good_text = self._reduction_search(initial_result, get_good_text)
            get_good_text = self._reduction_search(initial_result, get_good_text)
            get_good_text_result, search_over = self.get_goal_results([get_good_text])
            return get_good_text_result[0]

        if long_text_indices >= 11:  # 降低扰动率
            print("long_text_indices >= 11")
            get_good_text = self._reduction_search(initial_result, results[0].attacked_text)
            get_good_text_results0, search_over = self.get_goal_results([get_good_text])
            if search_over:
                return results[0]
            if get_good_text_results0[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                text_indices = initial_text.all_words_diff(get_good_text)
                text_indices = list(text_indices)
                text_indices.sort(key=None, reverse=False)
                long_text_indices = len(text_indices)
                if long_text_indices <= 10:
                    print("long_text_indices <= 10:", 2)
                    get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                    get_best_text_result, search_over = self.get_goal_results([get_best_text])
                    return get_best_text_result[0]
                if long_text_indices >= 11:
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text_results1, search_over = self.get_goal_results([get_good_text])
                    if search_over:
                        return get_good_text_results0[0]
                    if get_good_text_results1[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        text_indices = initial_text.all_words_diff(get_good_text)
                        text_indices = list(text_indices)
                        text_indices.sort(key=None, reverse=False)
                        long_text_indices = len(text_indices)
                        if long_text_indices <= 10:
                            print("long_text_indices <= 10:", 3)
                            get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                            get_best_text_result, search_over = self.get_goal_results([get_best_text])
                            return get_best_text_result[0]
                        if long_text_indices >= 11:
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text_results2, search_over = self.get_goal_results([get_good_text])
                            if search_over:
                                return get_good_text_results1[0]
                            if get_good_text_results2[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                text_indices = initial_text.all_words_diff(get_good_text)
                                text_indices = list(text_indices)
                                text_indices.sort(key=None, reverse=False)
                                long_text_indices = len(text_indices)
                                if long_text_indices <= 10:
                                    print("long_text_indices <= 10:", 4)
                                    get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                                    get_best_text_result, search_over = self.get_goal_results([get_best_text])
                                    return get_best_text_result[0]
                                if long_text_indices >= 11:
                                    get_good_text = self._reduction_search(initial_result, get_good_text)
                                    get_good_text_results3, search_over = self.get_goal_results([get_good_text])
                                    if search_over:
                                        return get_good_text_results2[0]
                                    if get_good_text_results3[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                        text_indices = initial_text.all_words_diff(get_good_text)
                                        text_indices = list(text_indices)
                                        text_indices.sort(key=None, reverse=False)
                                        long_text_indices = len(text_indices)
                                        if long_text_indices <= 10:
                                            print("long_text_indices <= 10:", 5)
                                            get_best_text = self._reduce_perturbation(initial_text, get_good_text)
                                            get_best_text_result4, search_over = self.get_goal_results(
                                                [get_best_text])
                                            return get_best_text_result4[0]
                                        if long_text_indices >= 11:
                                            get_good_text = self._reduction_search(initial_result,
                                                                                   get_good_text)
                                            get_good_text_results4, search_over = self.get_goal_results(
                                                [get_good_text])
                                            if search_over:
                                                return get_good_text_results3[0]
                                            if get_good_text_results4[
                                                0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                text_indices = initial_text.all_words_diff(get_good_text)
                                                text_indices = list(text_indices)
                                                text_indices.sort(key=None, reverse=False)
                                                long_text_indices = len(text_indices)
                                                if long_text_indices <= 10:
                                                    print("long_text_indices <= 10:", 6)
                                                    get_best_text = self._reduce_perturbation(initial_text,
                                                                                              get_good_text)
                                                    get_best_text_result5, search_over = self.get_goal_results(
                                                        [get_best_text])
                                                    return get_best_text_result5[0]
                                                if long_text_indices >= 11:
                                                    get_good_text = self._reduction_search(initial_result,
                                                                                           get_good_text)
                                                    get_good_text_results5, search_over = self.get_goal_results(
                                                        [get_good_text])
                                                    if search_over:
                                                        return get_good_text_results4[0]
                                                    if get_good_text_results5[
                                                        0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                        text_indices = initial_text.all_words_diff(get_good_text)
                                                        text_indices = list(text_indices)
                                                        text_indices.sort(key=None, reverse=False)
                                                        long_text_indices = len(text_indices)
                                                        if long_text_indices <= 10:
                                                            print("long_text_indices <= 10:", 7)
                                                            get_best_text = self._reduce_perturbation(initial_text,
                                                                                                      get_good_text)
                                                            get_best_text_result6, search_over = self.get_goal_results(
                                                                [get_best_text])
                                                            return get_best_text_result6[0]
                                                        if long_text_indices >= 11:
                                                            get_good_text = self._reduction_search(initial_result,
                                                                                                   get_good_text)
                                                            get_good_text_results6, search_over = self.get_goal_results(
                                                                [get_good_text])
                                                            if search_over:
                                                                return get_good_text_results5[0]
                                                            if get_good_text_results6[
                                                                0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                                text_indices = initial_text.all_words_diff(
                                                                    get_good_text)
                                                                text_indices = list(text_indices)
                                                                text_indices.sort(key=None, reverse=False)
                                                                long_text_indices = len(text_indices)
                                                                if long_text_indices <= 10:
                                                                    print("long_text_indices <= 10:", 8)
                                                                    get_best_text = self._reduce_perturbation(
                                                                        initial_text,
                                                                        get_good_text)
                                                                    get_best_text_result7, search_over = self.get_goal_results(
                                                                        [get_best_text])
                                                                    return get_best_text_result7[0]
                                                                else:
                                                                    get_best_text = self._multi_disturbance_text_reduce_perturbation(
                                                                        initial_text,
                                                                        get_good_text_results6[0].attacked_text)
                                                                    get_best_text_result8, search_over = self.get_goal_results(
                                                                        [get_best_text])
                                                                    return get_best_text_result8[0]

    def _level_1_optimization(self, population, initial_result):
        """
        Args:
            population([List[PopulationMember])  :the populations generated by Algorithmically
            initial_result (GoalFunctionResult): Original result.

        Returns:
            result as `AttackedText`
        """
        final_results_texts = []
        final_results_scores = []
        similarity_score = []
        pop_members = []
        best_result = defaultdict()
        for pop_member in population:
            similarity_score.append(pop_member.attributes["similarity_score"])
            pop_members.append(pop_member.attacked_text)
            imformation = list(zip(similarity_score, pop_members))
            imformation.sort(key=lambda score: score[0], reverse=True)
            final_text = imformation[0][1]
            final_similarity_score = imformation[0][0]
            result_pop_member = PopulationMember(
                final_text,
                attributes={
                    "similarity_score": final_similarity_score,
                    "valid_adversarial_example": True,
                },
            )
            print("level-1 get score", final_similarity_score)
            final_results_texts.append(final_text)
            if final_similarity_score >= 0.68:
                best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                return best_result[0]
            # 重新初始化
            initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)
            if initial_result.attacked_text.words == initial_adv_text.words:
                return initial_result
            new_text = self._reduction_search(initial_result, initial_adv_text)
            results, search_over = self.get_goal_results([new_text])
            if search_over:
                return init_adv_result
            if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                return results[0]
            initial_text = initial_result.attacked_text
            changed_indices = initial_text.all_words_diff(new_text)
            changed_indices = list(changed_indices)
            self._search_over = False
            population = self._initialize_populationDE(
                new_text, initial_text, changed_indices
            )

            if len(population) == 0:
                print("level-0")
                results0 = self._level_0_optimization(initial_text, initial_result, initial_adv_text)
                results0_text = results0.attacked_text
                final_results_texts.append(results0_text)
                for txt in final_results_texts:
                    txt_similarity_score = self._get_similarity_score(
                        [txt.text],
                        initial_text.text).tolist()
                    final_results_scores.append(txt_similarity_score)
                    imformation = list(zip(final_results_scores, final_results_texts))
                    imformation.sort(key=lambda score: score[0], reverse=True)
                get_text = imformation[0][1]
                results0, search_over = self.get_goal_results([get_text])
                print("w0")
                return results0[0]

            if len(changed_indices) == 1:
                print("level-1")
                similarity_score = []
                pop_members = []
                for pop_member in population:
                    similarity_score.append(pop_member.attributes["similarity_score"])
                    pop_members.append(pop_member.attacked_text)
                    imformation = list(zip(similarity_score, pop_members))
                    imformation.sort(key=lambda score: score[0], reverse=True)
                    final_text = imformation[0][1]
                    final_similarity_score = imformation[0][0]
                    result_pop_member = PopulationMember(
                        final_text,
                        attributes={
                            "similarity_score": final_similarity_score,
                            "valid_adversarial_example": True,
                        },
                    )
                    print("level-1 ? get", final_similarity_score)
                    if final_similarity_score >= 0.68:
                        best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                        return best_result[0]
                    else:
                        initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(
                            initial_result)
                        if initial_result.attacked_text.words == initial_adv_text.words:
                            return initial_result
                        new_text = self._reduction_search(initial_result, initial_adv_text)
                        results, search_over = self.get_goal_results([new_text])
                        if search_over:
                            return init_adv_result
                        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            return results[0]
                        initial_text = initial_result.attacked_text
                        changed_indices = initial_text.all_words_diff(new_text)
                        changed_indices = list(changed_indices)
                        self._search_over = False
                        population = self._initialize_populationDE(
                            new_text, initial_text, changed_indices
                        )

                        if len(population) == 0:
                            print("level-0")
                            results0 = self._level_0_optimization(initial_text, initial_result, initial_adv_text)
                            return results0
                        if len(changed_indices) == 1:
                            print("level-1")
                            similarity_score = []
                            pop_members = []
                            for pop_member in population:
                                similarity_score.append(pop_member.attributes["similarity_score"])
                                pop_members.append(pop_member.attacked_text)
                                imformation = list(zip(similarity_score, pop_members))
                                imformation.sort(key=lambda score: score[0], reverse=True)
                                final_text = imformation[0][1]
                                final_similarity_score = imformation[0][0]
                                result_pop_member = PopulationMember(
                                    final_text,
                                    attributes={
                                        "similarity_score": final_similarity_score,
                                        "valid_adversarial_example": True,
                                    },
                                )
                                print("level-1 ?? get score", final_similarity_score)
                                best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                                final_results_texts.append(final_text)
                                for txt in final_results_texts:
                                    txt_similarity_score = self._get_similarity_score(
                                        [txt.text],
                                        initial_text.text).tolist()
                                    final_results_scores.append(txt_similarity_score)
                                    imformation = list(zip(final_results_scores, final_results_texts))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                get_text = imformation[0][1]
                                best_result, search_over = self.get_goal_results([get_text])
                                print("w1")
                                return best_result[0]
                        if len(changed_indices) == 2:
                            print("level-2")

                            perturbed_text_text1 = []
                            perturbed_text_text2 = []
                            get_index1 = []
                            get_index2 = []
                            index1_to_word = []
                            index2_to_word = []
                            all_index = changed_indices
                            get_one_index = random.sample(changed_indices, 1)
                            for i in get_one_index:
                                index_1 = i
                                all_index.remove(i)
                            for j in all_index:
                                index_2 = j
                            for pop_members in population:
                                if pop_members.attributes["index"] == index_1:
                                    get_index1.append(index_1)
                                    index1_to_word.append(pop_members.attacked_text.words[index_1])
                                else:
                                    get_index2.append(index_2)
                                    index2_to_word.append(pop_members.attacked_text.words[index_2])

                            for pop_members in population:
                                if pop_members.attributes["index"] == index_1:
                                    new_text = pop_members.attacked_text.replace_words_at_indices(
                                        get_index2, index2_to_word
                                    )
                                    perturbed_text_text1.append(new_text)
                                    # print("perturbed_text_text1", perturbed_text_text1)
                                if pop_members.attributes["index"] == index_2:
                                    new_text = pop_members.attacked_text.replace_words_at_indices(
                                        get_index1, index1_to_word
                                    )
                                    perturbed_text_text2.append(new_text)
                                    # print("perturbed_text_text2", perturbed_text_text2)
                            select_text = perturbed_text_text1 + perturbed_text_text2
                            text_results, search_over = self.get_goal_results(select_text)
                            select_texts = []
                            for x in range(len(text_results)):
                                if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                    select_texts.append(text_results[x].attacked_text)
                            if len(select_texts) == 0:
                                return results[0]
                            txt_similarity_scores = []
                            for txt in select_texts:
                                txt_similarity_score = self._get_similarity_score(
                                    [txt.text],
                                    initial_text.text).tolist()
                                txt_similarity_scores.append(txt_similarity_score)
                            imformation = list(zip(txt_similarity_scores, select_texts))
                            imformation.sort(key=lambda score: score[0], reverse=True)
                            get_text = imformation[0][1]
                            get_similarity_score = float((imformation[0][0])[0])
                            get_indices = list(initial_text.all_words_diff(get_text))
                            final_texts = []
                            final_scores = []
                            for idx in get_indices:
                                final_text = get_text.replace_word_at_index(
                                    idx, initial_text.words[idx]
                                )
                                get_results, search_over = self.get_goal_results([final_text])

                                if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                    similarity_score = self._get_similarity_score(
                                        [final_text.text], initial_text.text
                                    ).tolist()[0]
                                    final_texts.append(final_text)
                                    final_scores.append(similarity_score)
                                    # print("final_texts", final_texts, "final_scores", final_scores)
                            if len(final_texts) >= 1:
                                final_information = list(zip(final_scores, final_texts))
                                final_information.sort(key=lambda score: score[0], reverse=True)
                                get_final_text = final_information[0][1]
                                get_final_score = float((final_information[0][0]))
                                result_pop_member = PopulationMember(
                                    get_final_text,
                                    attributes={
                                        "similarity_score": get_final_score,
                                        "valid_adversarial_example": True,
                                    },
                                )
                                print("level-2 ? get score", get_final_score)
                                final_results_texts.append(get_final_text)
                                for txt in final_results_texts:
                                    txt_similarity_score = self._get_similarity_score(
                                        [txt.text],
                                        initial_text.text).tolist()
                                    final_results_scores.append(txt_similarity_score)
                                    imformation = list(zip(final_results_scores, final_results_texts))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                get_text = imformation[0][1]
                                best_result, search_over = self.get_goal_results([get_text])
                                print("w2")
                                return best_result[0]

                            else:
                                similarity_score = []
                                pop_members = []
                                for pop_member in population:
                                    similarity_score.append(pop_member.attributes["similarity_score"])
                                    pop_members.append(pop_member.attacked_text)
                                    imformation = list(zip(similarity_score, pop_members))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                    final_text = imformation[0][1]
                                    final_score = imformation[0][0]
                                    result_pop_member = PopulationMember(
                                        final_text,
                                        attributes={
                                            "similarity_score": final_score,
                                            "valid_adversarial_example": True,
                                        },
                                    )
                                    print("level-2 ?? get score", final_score)
                                    if final_score >= 0.70:
                                        best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                                        return best_result[0]
                                    else:
                                        initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(
                                            initial_result)
                                        if initial_result.attacked_text.words == initial_adv_text.words:
                                            return initial_result
                                        new_text = self._reduction_search(initial_result, initial_adv_text)
                                        results, search_over = self.get_goal_results([new_text])
                                        if search_over:
                                            return init_adv_result
                                        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                                            return results[0]
                                        initial_text = initial_result.attacked_text
                                        changed_indices = initial_text.all_words_diff(new_text)
                                        changed_indices = list(changed_indices)
                                        self._search_over = False
                                        population = self._initialize_populationDE(
                                            new_text, initial_text, changed_indices
                                        )

                                        if len(population) == 0:
                                            print("level-0")
                                            results0 = self._level_0_optimization(initial_text, initial_result,
                                                                                  initial_adv_text)
                                            return results0
                                        if len(changed_indices) == 1:
                                            print("level-1")
                                            similarity_score = []
                                            pop_members = []
                                            for pop_member in population:
                                                similarity_score.append(pop_member.attributes["similarity_score"])
                                                pop_members.append(pop_member.attacked_text)
                                                imformation = list(zip(similarity_score, pop_members))
                                                imformation.sort(key=lambda score: score[0], reverse=True)
                                                final_text = imformation[0][1]
                                                final_similarity_score = imformation[0][0]
                                                result_pop_member = PopulationMember(
                                                    final_text,
                                                    attributes={
                                                        "similarity_score": final_similarity_score,
                                                        "valid_adversarial_example": True,
                                                    },
                                                )
                                                print("level-1 ??? get score", final_similarity_score)
                                                final_results_texts.append(final_text)
                                                for txt in final_results_texts:
                                                    txt_similarity_score = self._get_similarity_score(
                                                        [txt.text],
                                                        initial_text.text).tolist()
                                                    final_results_scores.append(txt_similarity_score)
                                                    imformation = list(zip(final_results_scores, final_results_texts))
                                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                                get_text = imformation[0][1]
                                                best_result, search_over = self.get_goal_results([get_text])
                                                print("w3")
                                                return best_result[0]
                                        if len(changed_indices) == 2:
                                            print("level-2")
                                            perturbed_text_text1 = []
                                            perturbed_text_text2 = []
                                            get_index1 = []
                                            get_index2 = []
                                            index1_to_word = []
                                            index2_to_word = []
                                            all_index = changed_indices
                                            get_one_index = random.sample(changed_indices, 1)
                                            for i in get_one_index:
                                                index_1 = i
                                                all_index.remove(i)
                                            for j in all_index:
                                                index_2 = j
                                            for pop_members in population:
                                                if pop_members.attributes["index"] == index_1:
                                                    get_index1.append(index_1)
                                                    index1_to_word.append(pop_members.attacked_text.words[index_1])
                                                else:
                                                    get_index2.append(index_2)
                                                    index2_to_word.append(pop_members.attacked_text.words[index_2])
                                            for pop_members in population:
                                                if pop_members.attributes["index"] == index_1:
                                                    new_text = pop_members.attacked_text.replace_words_at_indices(
                                                        get_index2, index2_to_word
                                                    )
                                                    perturbed_text_text1.append(new_text)

                                                if pop_members.attributes["index"] == index_2:
                                                    new_text = pop_members.attacked_text.replace_words_at_indices(
                                                        get_index1, index1_to_word
                                                    )
                                                    perturbed_text_text2.append(new_text)
                                            select_text = perturbed_text_text1 + perturbed_text_text2
                                            text_results, search_over = self.get_goal_results(select_text)
                                            select_texts = []
                                            for x in range(len(text_results)):
                                                if text_results[
                                                    x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                    select_texts.append(text_results[x].attacked_text)
                                            if len(select_texts) == 0:
                                                return results[0]
                                            txt_similarity_scores = []
                                            for txt in select_texts:
                                                txt_similarity_score = self._get_similarity_score(
                                                    [txt.text],
                                                    initial_text.text).tolist()
                                                txt_similarity_scores.append(txt_similarity_score)
                                            imformation = list(zip(txt_similarity_scores, select_texts))
                                            imformation.sort(key=lambda score: score[0], reverse=True)
                                            get_text = imformation[0][1]
                                            get_indices = list(initial_text.all_words_diff(get_text))
                                            final_texts = []
                                            final_scores = []
                                            for idx in get_indices:
                                                final_text = get_text.replace_word_at_index(
                                                    idx, initial_text.words[idx]
                                                )
                                                get_results, search_over = self.get_goal_results([final_text])
                                                if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                                    similarity_score = self._get_similarity_score(
                                                        [final_text.text], initial_text.text
                                                    ).tolist()[0]
                                                    final_texts.append(final_text)
                                                    final_scores.append(similarity_score)
                                            if len(final_texts) >= 1:
                                                final_information = list(zip(final_scores, final_texts))
                                                final_information.sort(key=lambda score: score[0], reverse=True)
                                                get_final_text = final_information[0][1]
                                                get_final_score = float((final_information[0][0]))
                                                result_pop_member = PopulationMember(
                                                    get_final_text,
                                                    attributes={
                                                        "similarity_score": get_final_score,
                                                        "valid_adversarial_example": True,
                                                    },
                                                )
                                                print("level-2????++ get score", get_final_score)
                                                final_results_texts.append(get_final_text)
                                                for txt in final_results_texts:
                                                    txt_similarity_score = self._get_similarity_score(
                                                        [txt.text],
                                                        initial_text.text).tolist()
                                                    final_results_scores.append(txt_similarity_score)
                                                    imformation = list(zip(final_results_scores, final_results_texts))
                                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                                get_text = imformation[0][1]
                                                best_result, search_over = self.get_goal_results([get_text])
                                                print("w4")
                                                return best_result[0]
                                            else:
                                                similarity_score = []
                                                pop_members = []
                                                for pop_member in population:
                                                    similarity_score.append(
                                                        pop_member.attributes["similarity_score"])
                                                    pop_members.append(pop_member.attacked_text)
                                                    imformation = list(zip(similarity_score, pop_members))
                                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                                    final_text = imformation[0][1]
                                                    final_score = imformation[0][0]
                                                    result_pop_member = PopulationMember(
                                                        final_text,
                                                        attributes={
                                                            "similarity_score": final_score,
                                                            "valid_adversarial_example": True,
                                                        },
                                                    )
                                                    print("level-2 ?????+ get score", final_score)
                                                    final_results_texts.append(final_text)
                                                    for txt in final_results_texts:
                                                        txt_similarity_score = self._get_similarity_score(
                                                            [txt.text],
                                                            initial_text.text).tolist()
                                                        final_results_scores.append(txt_similarity_score)
                                                        imformation = list(
                                                            zip(final_results_scores, final_results_texts))
                                                        imformation.sort(key=lambda score: score[0], reverse=True)
                                                    get_text = imformation[0][1]
                                                    best_result, search_over = self.get_goal_results([get_text])
                                                    print("w5")
                                                    return best_result[0]
                                        if len(changed_indices) >= 3:  # 这边还需要替换成最终的结果
                                            print("level-3")
                                            similarity_score = []
                                            pop_members = []
                                            for pop_member in population:
                                                similarity_score.append(pop_member.attributes["similarity_score"])
                                                pop_members.append(pop_member.attacked_text)
                                                imformation = list(zip(similarity_score, pop_members))
                                                imformation.sort(key=lambda score: score[0], reverse=True)
                                                final_text = imformation[0][1]
                                                final_score = imformation[0][0]
                                                result_pop_member = PopulationMember(
                                                    final_text,
                                                    attributes={
                                                        "similarity_score": final_score,
                                                        "valid_adversarial_example": True,
                                                    },
                                                )
                                                print("mid_score", final_score)
                                                best_result, _ = self.get_goal_results(
                                                    [result_pop_member.attacked_text])
                                                print("w6")
                                                return best_result[0]

            if len(changed_indices) == 2:
                final_results_texts = []
                final_results_scores = []
                perturbed_text_text1 = []
                perturbed_text_text2 = []
                get_index1 = []
                get_index2 = []
                index1_to_word = []
                index2_to_word = []
                all_index = changed_indices
                get_one_index = random.sample(changed_indices, 1)
                for i in get_one_index:
                    index_1 = i
                    all_index.remove(i)
                for j in all_index:
                    index_2 = j
                for pop_members in population:
                    if pop_members.attributes["index"] == index_1:
                        get_index1.append(index_1)
                        index1_to_word.append(pop_members.attacked_text.words[index_1])
                    else:
                        get_index2.append(index_2)
                        index2_to_word.append(pop_members.attacked_text.words[index_2])

                for pop_members in population:
                    if pop_members.attributes["index"] == index_1:
                        new_text = pop_members.attacked_text.replace_words_at_indices(
                            get_index2, index2_to_word
                        )
                        perturbed_text_text1.append(new_text)
                        # print("perturbed_text_text1", perturbed_text_text1)
                    if pop_members.attributes["index"] == index_2:
                        new_text = pop_members.attacked_text.replace_words_at_indices(
                            get_index1, index1_to_word
                        )
                        perturbed_text_text2.append(new_text)
                        # print("perturbed_text_text2", perturbed_text_text2)
                select_text = perturbed_text_text1 + perturbed_text_text2
                text_results, search_over = self.get_goal_results(select_text)
                select_texts = []
                for x in range(len(text_results)):
                    if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        select_texts.append(text_results[x].attacked_text)
                if len(select_texts) == 0:
                    return results[0]
                txt_similarity_scores = []
                for txt in select_texts:
                    txt_similarity_score = self._get_similarity_score(
                        [txt.text],
                        initial_text.text).tolist()
                    txt_similarity_scores.append(txt_similarity_score)
                imformation = list(zip(txt_similarity_scores, select_texts))
                imformation.sort(key=lambda score: score[0], reverse=True)
                get_text = imformation[0][1]
                get_similarity_score = float((imformation[0][0])[0])
                get_indices = list(initial_text.all_words_diff(get_text))
                final_texts = []
                final_scores = []
                for idx in get_indices:
                    final_text = get_text.replace_word_at_index(
                        idx, initial_text.words[idx]
                    )
                    get_results, search_over = self.get_goal_results([final_text])

                    if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        similarity_score = self._get_similarity_score(
                            [final_text.text], initial_text.text
                        ).tolist()[0]
                        final_texts.append(final_text)
                        final_scores.append(similarity_score)
                        # print("final_texts", final_texts, "final_scores", final_scores)
                if len(final_texts) >= 1:
                    final_information = list(zip(final_scores, final_texts))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                    get_final_score = float((final_information[0][0]))
                    result_pop_member = PopulationMember(
                        get_final_text,
                        attributes={
                            "similarity_score": get_final_score,
                            "valid_adversarial_example": True,
                        },
                    )
                    print("levle-2 get ?+ final score", get_final_score)
                    best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                    return best_result[0]
                else:
                    similarity_score = []
                    pop_members = []
                    for pop_member in population:
                        similarity_score.append(pop_member.attributes["similarity_score"])
                        pop_members.append(pop_member.attacked_text)
                        imformation = list(zip(similarity_score, pop_members))
                        imformation.sort(key=lambda score: score[0], reverse=True)
                        final_text = imformation[0][1]
                        final_score = imformation[0][0]
                        result_pop_member = PopulationMember(
                            final_text,
                            attributes={
                                "similarity_score": final_score,
                                "valid_adversarial_example": True,
                            },
                        )
                        print("level-2 +++? get score", final_score)
                        if final_score >= 0.70:
                            best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                            return best_result[0]
                        else:
                            final_results_texts.append(final_text)
                            initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(
                                initial_result)
                            if initial_result.attacked_text.words == initial_adv_text.words:
                                return initial_result
                            new_text = self._reduction_search(initial_result, initial_adv_text)
                            results, search_over = self.get_goal_results([new_text])
                            if search_over:
                                return init_adv_result
                            if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                                return results[0]
                            initial_text = initial_result.attacked_text
                            changed_indices = initial_text.all_words_diff(new_text)
                            changed_indices = list(changed_indices)
                            self._search_over = False
                            population = self._initialize_populationDE(
                                new_text, initial_text, changed_indices
                            )

                            if len(population) == 0:
                                print("level-0")
                                results0 = self._level_0_optimization(initial_text, initial_result, initial_adv_text)
                                return results0[0]
                            if len(changed_indices) == 1:
                                print("level-1")
                                similarity_score = []
                                pop_members = []
                                for pop_member in population:
                                    similarity_score.append(pop_member.attributes["similarity_score"])
                                    pop_members.append(pop_member.attacked_text)
                                    imformation = list(zip(similarity_score, pop_members))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                    final_text = imformation[0][1]
                                    final_similarity_score = imformation[0][0]
                                    result_pop_member = PopulationMember(
                                        final_text,
                                        attributes={
                                            "similarity_score": final_similarity_score,
                                            "valid_adversarial_example": True,
                                        },
                                    )
                                    print("level-1 !++ get score", final_similarity_score)
                                    final_results_texts.append(final_text)
                                    for txt in final_results_texts:
                                        txt_similarity_score = self._get_similarity_score(
                                            [txt.text],
                                            initial_text.text).tolist()
                                        final_results_scores.append(txt_similarity_score)
                                        imformation = list(
                                            zip(final_results_scores, final_results_texts))
                                        imformation.sort(key=lambda score: score[0], reverse=True)
                                    get_text = imformation[0][1]
                                    best_result, search_over = self.get_goal_results([get_text])
                                    print("w7")
                                    return best_result[0]

                            if len(changed_indices) == 2:
                                print("level-2")
                                perturbed_text_text1 = []
                                perturbed_text_text2 = []
                                get_index1 = []
                                get_index2 = []
                                index1_to_word = []
                                index2_to_word = []
                                all_index = changed_indices
                                get_one_index = random.sample(changed_indices, 1)
                                for i in get_one_index:
                                    index_1 = i
                                    all_index.remove(i)
                                for j in all_index:
                                    index_2 = j
                                for pop_members in population:
                                    if pop_members.attributes["index"] == index_1:
                                        get_index1.append(index_1)
                                        index1_to_word.append(pop_members.attacked_text.words[index_1])
                                    else:
                                        get_index2.append(index_2)
                                        index2_to_word.append(pop_members.attacked_text.words[index_2])
                                for pop_members in population:
                                    if pop_members.attributes["index"] == index_1:
                                        new_text = pop_members.attacked_text.replace_words_at_indices(
                                            get_index2, index2_to_word
                                        )
                                        perturbed_text_text1.append(new_text)

                                    if pop_members.attributes["index"] == index_2:
                                        new_text = pop_members.attacked_text.replace_words_at_indices(
                                            get_index1, index1_to_word
                                        )
                                        perturbed_text_text2.append(new_text)
                                select_text = perturbed_text_text1 + perturbed_text_text2
                                text_results, search_over = self.get_goal_results(select_text)
                                select_texts = []
                                for x in range(len(text_results)):
                                    if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                        select_texts.append(text_results[x].attacked_text)
                                if len(select_texts) == 0:
                                    return results[0]
                                txt_similarity_scores = []
                                for txt in select_texts:
                                    txt_similarity_score = self._get_similarity_score(
                                        [txt.text],
                                        initial_text.text).tolist()
                                    txt_similarity_scores.append(txt_similarity_score)
                                imformation = list(zip(txt_similarity_scores, select_texts))
                                imformation.sort(key=lambda score: score[0], reverse=True)
                                get_text = imformation[0][1]
                                get_indices = list(initial_text.all_words_diff(get_text))
                                final_texts = []
                                final_scores = []
                                for idx in get_indices:
                                    final_text = get_text.replace_word_at_index(
                                        idx, initial_text.words[idx]
                                    )
                                    get_results, search_over = self.get_goal_results([final_text])
                                    if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                        similarity_score = self._get_similarity_score(
                                            [final_text.text], initial_text.text
                                        ).tolist()[0]
                                        final_texts.append(final_text)
                                        final_scores.append(similarity_score)
                                if len(final_texts) >= 1:
                                    final_information = list(zip(final_scores, final_texts))
                                    final_information.sort(key=lambda score: score[0], reverse=True)
                                    get_final_text = final_information[0][1]
                                    get_final_score = float((final_information[0][0]))
                                    result_pop_member = PopulationMember(
                                        get_final_text,
                                        attributes={
                                            "similarity_score": get_final_score,
                                            "valid_adversarial_example": True,
                                        },
                                    )
                                    print("level-2 ?+++-+ get score", get_final_score)
                                    final_results_texts.append(get_final_text)
                                    for txt in final_results_texts:
                                        txt_similarity_score = self._get_similarity_score(
                                            [txt.text],
                                            initial_text.text).tolist()
                                        final_results_scores.append(txt_similarity_score)
                                        imformation = list(
                                            zip(final_results_scores, final_results_texts))
                                        imformation.sort(key=lambda score: score[0], reverse=True)
                                    get_text = imformation[0][1]
                                    best_result, search_over = self.get_goal_results([get_text])
                                    print("w8")
                                    return best_result[0]

                                else:
                                    similarity_score = []
                                    pop_members = []
                                    for pop_member in population:
                                        similarity_score.append(pop_member.attributes["similarity_score"])
                                        pop_members.append(pop_member.attacked_text)
                                        imformation = list(zip(similarity_score, pop_members))
                                        imformation.sort(key=lambda score: score[0], reverse=True)
                                        final_text = imformation[0][1]
                                        final_score = imformation[0][0]
                                        result_pop_member = PopulationMember(
                                            final_text,
                                            attributes={
                                                "similarity_score": final_score,
                                                "valid_adversarial_example": True,
                                            },
                                        )

                                        print("level-2 !+???+ get score", final_score)
                                        final_results_texts.append(final_text)
                                        for txt in final_results_texts:
                                            txt_similarity_score = self._get_similarity_score(
                                                [txt.text],
                                                initial_text.text).tolist()
                                            final_results_scores.append(txt_similarity_score)
                                            imformation = list(zip(final_results_scores, final_results_texts))
                                            imformation.sort(key=lambda score: score[0], reverse=True)
                                        get_text = imformation[0][1]
                                        print("w9")
                                        best_result, _ = self.get_goal_results([get_text])
                                        return best_result[0]
                            if len(changed_indices) >= 3:  # 这边还需要替换成最终的结果
                                print("level-3")
                                similarity_score = []
                                pop_members = []
                                for pop_member in population:
                                    similarity_score.append(pop_member.attributes["similarity_score"])
                                    pop_members.append(pop_member.attacked_text)
                                    imformation = list(zip(similarity_score, pop_members))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                    final_text = imformation[0][1]
                                    final_score = imformation[0][0]
                                    result_pop_member = PopulationMember(
                                        final_text,
                                        attributes={
                                            "similarity_score": final_score,
                                            "valid_adversarial_example": True,
                                        },
                                    )
                                    print("mid_score+", final_score)
                                    print("w10")
                                    best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                                    return best_result[0]

    def _level_2_optimization(self, changed_indices, population, initial_text, initial_result, results):
        """
        Args:
            changed_indices (list[int]): The indices changed in the `cur_adv_text` from the `initial_text`.
            population([List[PopulationMember])  :the populations generated by Algorithmically
            initial_text (AttackedText): The original input text.
            initial_text (AttackedText): The original input text.
            initial_result (GoalFunctionResult): Original result.
            results(GoalFunctionResult) :the result through Search space reduction
        Returns:
            result as `AttackedText`
        """
        final_results_texts = []
        final_results_scores = []
        perturbed_text_text1 = []
        perturbed_text_text2 = []
        get_index1 = []
        get_index2 = []
        index1_to_word = []
        index2_to_word = []
        all_index = changed_indices
        get_one_index = random.sample(changed_indices, 1)
        for i in get_one_index:
            index_1 = i
            all_index.remove(i)
        for j in all_index:
            index_2 = j
        for pop_members in population:
            if pop_members.attributes["index"] == index_1:
                get_index1.append(index_1)
                index1_to_word.append(pop_members.attacked_text.words[index_1])
            else:
                get_index2.append(index_2)
                index2_to_word.append(pop_members.attacked_text.words[index_2])

        for pop_members in population:
            if pop_members.attributes["index"] == index_1:
                new_text = pop_members.attacked_text.replace_words_at_indices(
                    get_index2, index2_to_word
                )
                perturbed_text_text1.append(new_text)
                # print("perturbed_text_text1", perturbed_text_text1)
            if pop_members.attributes["index"] == index_2:
                new_text = pop_members.attacked_text.replace_words_at_indices(
                    get_index1, index1_to_word
                )
                perturbed_text_text2.append(new_text)
                # print("perturbed_text_text2", perturbed_text_text2)
        select_text = perturbed_text_text1 + perturbed_text_text2
        text_results, search_over = self.get_goal_results(select_text)
        select_texts = []
        for x in range(len(text_results)):
            if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                select_texts.append(text_results[x].attacked_text)
        if len(select_texts) == 0:
            return results[0]
        txt_similarity_scores = []
        for txt in select_texts:
            txt_similarity_score = self._get_similarity_score(
                [txt.text],
                initial_text.text).tolist()
            txt_similarity_scores.append(txt_similarity_score)
        imformation = list(zip(txt_similarity_scores, select_texts))
        imformation.sort(key=lambda score: score[0], reverse=True)
        get_text = imformation[0][1]
        get_similarity_score = float((imformation[0][0])[0])
        get_indices = list(initial_text.all_words_diff(get_text))
        final_texts = []
        final_scores = []
        for idx in get_indices:
            final_text = get_text.replace_word_at_index(
                idx, initial_text.words[idx]
            )
            get_results, search_over = self.get_goal_results([final_text])

            if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                similarity_score = self._get_similarity_score(
                    [final_text.text], initial_text.text
                ).tolist()[0]
                final_texts.append(final_text)
                final_scores.append(similarity_score)
                # print("final_texts", final_texts, "final_scores", final_scores)
        if len(final_texts) >= 1:
            final_information = list(zip(final_scores, final_texts))
            final_information.sort(key=lambda score: score[0], reverse=True)
            get_final_text = final_information[0][1]
            get_final_score = float((final_information[0][0]))
            result_pop_member = PopulationMember(
                get_final_text,
                attributes={
                    "similarity_score": get_final_score,
                    "valid_adversarial_example": True,
                },
            )
            print("levle-2 get ?+ final score", get_final_score)
            best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
            return best_result[0]
        else:
            similarity_score = []
            pop_members = []
            for pop_member in population:
                similarity_score.append(pop_member.attributes["similarity_score"])
                pop_members.append(pop_member.attacked_text)
                imformation = list(zip(similarity_score, pop_members))
                imformation.sort(key=lambda score: score[0], reverse=True)
                final_text = imformation[0][1]
                final_score = imformation[0][0]
                result_pop_member = PopulationMember(
                    final_text,
                    attributes={
                        "similarity_score": final_score,
                        "valid_adversarial_example": True,
                    },
                )
                print("level-2 +++? get score", final_score)
                if final_score >= 0.70:
                    best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                    return best_result[0]
                else:
                    final_results_texts.append(final_text)
                    initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)
                    if initial_result.attacked_text.words == initial_adv_text.words:
                        return initial_result
                    new_text = self._reduction_search(initial_result, initial_adv_text)
                    results, search_over = self.get_goal_results([new_text])
                    if search_over:
                        return init_adv_result
                    if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        return results[0]
                    initial_text = initial_result.attacked_text
                    changed_indices = initial_text.all_words_diff(new_text)
                    changed_indices = list(changed_indices)
                    self._search_over = False
                    population = self._initialize_populationDE(
                        new_text, initial_text, changed_indices
                    )

                    if len(population) == 0:
                        print("level-0")
                        results0 = self._level_0_optimization(initial_text, initial_result, initial_adv_text)
                        return results0[0]
                    if len(changed_indices) == 1:
                        print("level-1")
                        similarity_score = []
                        pop_members = []
                        for pop_member in population:
                            similarity_score.append(pop_member.attributes["similarity_score"])
                            pop_members.append(pop_member.attacked_text)
                            imformation = list(zip(similarity_score, pop_members))
                            imformation.sort(key=lambda score: score[0], reverse=True)
                            final_text = imformation[0][1]
                            final_similarity_score = imformation[0][0]
                            result_pop_member = PopulationMember(
                                final_text,
                                attributes={
                                    "similarity_score": final_similarity_score,
                                    "valid_adversarial_example": True,
                                },
                            )
                            print("level-1 !++ get score", final_similarity_score)
                            final_results_texts.append(final_text)
                            for txt in final_results_texts:
                                txt_similarity_score = self._get_similarity_score(
                                    [txt.text],
                                    initial_text.text).tolist()
                                final_results_scores.append(txt_similarity_score)
                                imformation = list(
                                    zip(final_results_scores, final_results_texts))
                                imformation.sort(key=lambda score: score[0], reverse=True)
                            get_text = imformation[0][1]
                            best_result, search_over = self.get_goal_results([get_text])
                            print("w7")
                            return best_result[0]

                    if len(changed_indices) == 2:
                        print("level-2")
                        perturbed_text_text1 = []
                        perturbed_text_text2 = []
                        get_index1 = []
                        get_index2 = []
                        index1_to_word = []
                        index2_to_word = []
                        all_index = changed_indices
                        get_one_index = random.sample(changed_indices, 1)
                        for i in get_one_index:
                            index_1 = i
                            all_index.remove(i)
                        for j in all_index:
                            index_2 = j
                        for pop_members in population:
                            if pop_members.attributes["index"] == index_1:
                                get_index1.append(index_1)
                                index1_to_word.append(pop_members.attacked_text.words[index_1])
                            else:
                                get_index2.append(index_2)
                                index2_to_word.append(pop_members.attacked_text.words[index_2])
                        for pop_members in population:
                            if pop_members.attributes["index"] == index_1:
                                new_text = pop_members.attacked_text.replace_words_at_indices(
                                    get_index2, index2_to_word
                                )
                                perturbed_text_text1.append(new_text)

                            if pop_members.attributes["index"] == index_2:
                                new_text = pop_members.attacked_text.replace_words_at_indices(
                                    get_index1, index1_to_word
                                )
                                perturbed_text_text2.append(new_text)
                        select_text = perturbed_text_text1 + perturbed_text_text2
                        text_results, search_over = self.get_goal_results(select_text)
                        select_texts = []
                        for x in range(len(text_results)):
                            if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                select_texts.append(text_results[x].attacked_text)
                        if len(select_texts) == 0:
                            return results[0]
                        txt_similarity_scores = []
                        for txt in select_texts:
                            txt_similarity_score = self._get_similarity_score(
                                [txt.text],
                                initial_text.text).tolist()
                            txt_similarity_scores.append(txt_similarity_score)
                        imformation = list(zip(txt_similarity_scores, select_texts))
                        imformation.sort(key=lambda score: score[0], reverse=True)
                        get_text = imformation[0][1]
                        get_indices = list(initial_text.all_words_diff(get_text))
                        final_texts = []
                        final_scores = []
                        for idx in get_indices:
                            final_text = get_text.replace_word_at_index(
                                idx, initial_text.words[idx]
                            )
                            get_results, search_over = self.get_goal_results([final_text])
                            if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                similarity_score = self._get_similarity_score(
                                    [final_text.text], initial_text.text
                                ).tolist()[0]
                                final_texts.append(final_text)
                                final_scores.append(similarity_score)
                        if len(final_texts) >= 1:
                            final_information = list(zip(final_scores, final_texts))
                            final_information.sort(key=lambda score: score[0], reverse=True)
                            get_final_text = final_information[0][1]
                            get_final_score = float((final_information[0][0]))
                            result_pop_member = PopulationMember(
                                get_final_text,
                                attributes={
                                    "similarity_score": get_final_score,
                                    "valid_adversarial_example": True,
                                },
                            )
                            print("level-2 ?+++-+ get score", get_final_score)
                            final_results_texts.append(get_final_text)
                            for txt in final_results_texts:
                                txt_similarity_score = self._get_similarity_score(
                                    [txt.text],
                                    initial_text.text).tolist()
                                final_results_scores.append(txt_similarity_score)
                                imformation = list(
                                    zip(final_results_scores, final_results_texts))
                                imformation.sort(key=lambda score: score[0], reverse=True)
                            get_text = imformation[0][1]
                            best_result, search_over = self.get_goal_results([get_text])
                            print("w8")
                            return best_result[0]

                        else:
                            similarity_score = []
                            pop_members = []
                            for pop_member in population:
                                similarity_score.append(pop_member.attributes["similarity_score"])
                                pop_members.append(pop_member.attacked_text)
                                imformation = list(zip(similarity_score, pop_members))
                                imformation.sort(key=lambda score: score[0], reverse=True)
                                final_text = imformation[0][1]
                                final_score = imformation[0][0]
                                result_pop_member = PopulationMember(
                                    final_text,
                                    attributes={
                                        "similarity_score": final_score,
                                        "valid_adversarial_example": True,
                                    },
                                )

                                print("level-2 !+???+ get score", final_score)
                                final_results_texts.append(final_text)
                                for txt in final_results_texts:
                                    txt_similarity_score = self._get_similarity_score(
                                        [txt.text],
                                        initial_text.text).tolist()
                                    final_results_scores.append(txt_similarity_score)
                                    imformation = list(zip(final_results_scores, final_results_texts))
                                    imformation.sort(key=lambda score: score[0], reverse=True)
                                get_text = imformation[0][1]
                                print("w9")
                                best_result, _ = self.get_goal_results([get_text])
                                return best_result[0]
                    if len(changed_indices) >= 3:  # 这边还需要替换成最终的结果
                        print("level-3")
                        similarity_score = []
                        pop_members = []
                        for pop_member in population:
                            similarity_score.append(pop_member.attributes["similarity_score"])
                            pop_members.append(pop_member.attacked_text)
                            imformation = list(zip(similarity_score, pop_members))
                            imformation.sort(key=lambda score: score[0], reverse=True)
                            final_text = imformation[0][1]
                            final_score = imformation[0][0]
                            result_pop_member = PopulationMember(
                                final_text,
                                attributes={
                                    "similarity_score": final_score,
                                    "valid_adversarial_example": True,
                                },
                            )
                            print("mid_score+", final_score)
                            print("w10")
                            best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                            return best_result[0]

    def _level_3_optimization(self, initial_text, initial_result, population, new_text, results):
        """
        Args:
            initial_text (AttackedText): The original input text.
            initial_result (GoalFunctionResult): Original result.
            population([List[PopulationMember])  :the populations generated by Algorithmically
            new_text (AttackedText):the text through Search space reduction
            results(GoalFunctionResult) :the result through Search space reduction
        Returns:
            result as `AttackedText`
        """
        index_to_replace = []
        words_to_replace = []
        scores_to_replace = []
        changed_indices = initial_text.all_words_diff(new_text)
        changed_indices = list(changed_indices)
        # print("??changed_indices", changed_indices)
        for idx in changed_indices:
            for pop_members in population:
                if pop_members.attributes["index"] == idx:

                    index_to_replace.append(idx)
                    words_to_replace.append(pop_members.attacked_text.words[idx])
                    if pop_members.attributes["index"] == idx \
                            and pop_members.attributes["index_to_word"] == pop_members.attacked_text.words[idx]:
                        scores_to_replace.append(pop_members.attributes["similarity_score"])
        index_to_replace.reverse()
        words_to_replace.reverse()
        scores_to_replace.reverse()
        # print("index_to_replace", index_to_replace, "words_to_replace", words_to_replace, "scores_to_replace",
        #       scores_to_replace)

        similarity_scores = []
        pop_members = []
        good_pop_member = defaultdict
        good_pop_member_word = defaultdict
        for pop_member in population:
            similarity_scores.append(pop_member.attributes["similarity_score"])
            pop_members.append(pop_member.attacked_text)
            imformation = list(zip(similarity_scores, pop_members))
            imformation.sort(key=lambda score: score[0], reverse=True)
            final_text = imformation[0][1]
            final_score = imformation[0][0]
            good_pop_member = PopulationMember(
                final_text,
                attributes={
                    "similarity_score": final_score,
                    "valid_adversarial_example": True,
                }
            )
            # 选择语义相似度最高的种群，优化,将可以替换的位置，都替换成语义最高的单词
        mid_text = good_pop_member.attacked_text.replace_words_at_indices(
            index_to_replace, words_to_replace
        )
        mid_text_result, search_over = self.get_goal_results([mid_text])
        if mid_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
            print("ok,go on")
            good_pop_member_indice = results[0].attacked_text.all_words_diff(good_pop_member.attacked_text)
            good_pop_member_indice = list(good_pop_member_indice)

            # 调整原始位置的单词，此处索引全部为“1”
            if len(good_pop_member_indice) >= 1:
                good_text = defaultdict  # good_text是目前，将扰动位置都换成了，最高语义的单词的对抗样本，扰动率高
                for i in good_pop_member_indice:
                    good_pop_member_word = good_pop_member.attacked_text.words[i]
                    good_text = mid_text.replace_words_at_indices(
                        good_pop_member_indice, [good_pop_member_word]
                    )
                good_text_indices = initial_text.all_words_diff(good_text)
                good_text_indices = list(good_text_indices)
                good_text_indices.sort(key=None, reverse=False)
                long_good_text_indices = len(good_text_indices)
                get_good_texts = []
                a = initial_text.text
                if 1 <= long_good_text_indices <= 10:
                    print("1 <= long_good_text_indices <= 10")
                    # result = self._direct_optimization(initial_text, initial_result, good_text)
                    get_good_text = self._reduce_perturbation(initial_text, good_text)
                    get_good_texts.append(get_good_text)

                if 11 <= long_good_text_indices <= 14:
                    print("11 <= long_good_text_indices <= 14")
                    get_good_text = self._reduction_search(initial_result, good_text)
                    good_text_indice = initial_text.all_words_diff(get_good_text)
                    good_text_indice = list(good_text_indice)
                    long_good_text_indice = len(good_text_indice)
                    if 1 <= long_good_text_indice <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._reduction_search(initial_result, good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        get_good_texts.append(get_good_text)

                if 15 <= long_good_text_indices <= 20:
                    print("15 <= long_good_text_indices <= 20")

                    get_good_text = self._reduction_search(initial_result, good_text)
                    good_text_indice = initial_text.all_words_diff(get_good_text)
                    good_text_indice = list(good_text_indice)
                    long_good_text_indice = len(good_text_indice)
                    print("long_good_text_indice", long_good_text_indice)
                    if 1 <= long_good_text_indice <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._reduction_search(initial_result, good_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        get_good_texts.append(get_good_text)

                if 21 <= long_good_text_indices <= 30:
                    print("21 <= long_good_text_indices <= 30")

                    get_good_text = self._reduction_search(initial_result, good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    good_text_indice = initial_text.all_words_diff(get_good_text)
                    good_text_indice = list(good_text_indice)
                    long_good_text_indice = len(good_text_indice)
                    if 1 <= long_good_text_indice <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        get_good_texts.append(get_good_text)

                if 31 <= long_good_text_indices <= 50:
                    print("31 <= long_good_text_indices <= 40")
                    get_good_text = self._reduction_search(initial_result, good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                    good_text_indice = initial_text.all_words_diff(get_good_text)
                    good_text_indice = list(good_text_indice)
                    long_good_text_indice = len(good_text_indice)
                    if 1 <= long_good_text_indice <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        get_good_texts.append(get_good_text)

                if 50 <= long_good_text_indices <= 100:
                    print("50 <= long_good_text_indices <= 100")
                    get_good_text = self._reduction_search(initial_result, good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    good_text_indice = initial_text.all_words_diff(get_good_text)
                    good_text_indice = list(good_text_indice)
                    long_good_text_indice = len(good_text_indice)
                    if 1 <= long_good_text_indice <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        get_good_texts.append(get_good_text)

                elif long_good_text_indices >= 100:
                    print("long_good_text_indices >= 100")
                    result = self._direct_optimization(initial_text, initial_result, good_text)
                    result = self._direct_optimization(initial_text, initial_result, result.attacked_text)
                    get_indice = initial_text.all_words_diff(result.attacked_text)
                    get_indice = list(get_indice)
                    if len(get_indice) <= 10:
                        get_good_text = self._reduce_perturbation(initial_text, result.attacked_text)
                    elif len(get_indice) <= 30:
                        get_good_text = self._reduction_search(initial_result, result.attacked_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_texts.append(get_good_text)
                    else:
                        get_good_text = self._multi_disturbance_text_reduce_perturbation(initial_text,
                                                                                         result.attacked_text)
                    get_good_texts.append(get_good_text)
                # elif 101 <= long_good_text_indices <= 150:
                #     print("101 <= long_good_text_indices <= 150")
                #     get_good_text = self._reduce_perturbation(initial_text, good_text)
                #     get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                #     get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                #     get_good_texts.append(get_good_text)
                # elif 151 <= long_good_text_indices <= 200:
                #     get_good_text = self._reduce_perturbation(initial_text, good_text)
                #     get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                #     get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                #     get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                #     get_good_texts.append(get_good_text)

                indexs_test = []
                words_test = []
                for i in good_text_indices:
                    test_text3 = good_text.replace_words_at_indices(
                        [i], [initial_text.words[i]]
                    )
                    test_text_result, search_over = self.get_goal_results([test_text3])
                    if test_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        indexs_test.append(i)
                        words_test.append(initial_text.words[i])
                        test_text4 = good_text.replace_words_at_indices(
                            indexs_test, words_test
                        )
                        test_text_result, search_over = self.get_goal_results([test_text4])
                        if test_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            get_good_texts.append(test_text_result[0].attacked_text)
                final_good_scores = []
                get_final_text = defaultdict
                for txt in get_good_texts:
                    similarity_score = self._get_similarity_score(
                        [txt.text], initial_text.text
                    ).tolist()[0]
                    final_good_scores.append(similarity_score)
                    final_information = list(zip(final_good_scores, get_good_texts))
                    final_information.sort(key=lambda score: score[0], reverse=True)
                    get_final_text = final_information[0][1]
                best_text_results, search_over = self.get_goal_results([get_final_text])
                if best_text_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return best_text_results[0]
                else:
                    best_result, _ = self.get_goal_results([good_text])
                    return best_result[0]
            else:
                best_result, _ = self.get_goal_results([mid_text])
                return best_result[0]
                # good_text = mid_text
                # 将每个索引位置的单词，替换为语义相似度最高的单词
                # 降低扰动率

        else:
            print("no,back")  # 这边的效果不好，需要逐一替换
            a = initial_text.text
            print(a)

            similarity_score = []
            pop_members = []
            final_text = defaultdict
            result_pop_member = defaultdict
            for pop_member in population:
                similarity_score.append(pop_member.attributes["similarity_score"])
                pop_members.append(pop_member.attacked_text)
                imformation = list(zip(similarity_score, pop_members))
                imformation.sort(key=lambda score: score[0], reverse=True)
                final_text = imformation[0][1]
                final_score = imformation[0][0]
                result_pop_member = PopulationMember(
                    final_text,
                    attributes={
                        "similarity_score": final_score,
                        "valid_adversarial_example": True,
                    },
                )
            good_texts_scores2 = []
            good_texts_test2 = []
            good_texts_test3 = []

            replacements = list(zip(index_to_replace, words_to_replace))
            for i, word in replacements:
                test_text1 = final_text.replace_words_at_indices(
                    [i], [word]
                )
                test_text_result1, search_over = self.get_goal_results([test_text1])
                if test_text_result1[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:

                    good_texts_test2.append(test_text_result1[0].attacked_text)
                    for txt in good_texts_test2:
                        txt_similarity_score = self._get_similarity_score(
                            [txt.text],
                            initial_text.text).tolist()
                        good_texts_scores2.append(txt_similarity_score)
            imformation = list(zip(good_texts_scores2, good_texts_test2))
            imformation.sort(key=lambda score: score[0], reverse=True)
            scores_1, txts_1 = zip(*imformation)
            txts_1 = list(txts_1)
            if len(txts_1) == 1:
                good_texts_test3.append(txts_1[0])
            if len(txts_1) == 2:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[1])
            if len(txts_1) == 3:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[2])
            if len(txts_1) == 4:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[2])
            if len(txts_1) == 5:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[2])
                good_texts_test3.append(txts_1[4])
            if 6 <= len(txts_1) <= 7:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[2])
                good_texts_test3.append(txts_1[5])
            if len(txts_1) >= 8:
                good_texts_test3.append(txts_1[0])
                good_texts_test3.append(txts_1[2])
                good_texts_test3.append(txts_1[4])
                good_texts_test3.append(txts_1[7])
            get_best_texts = []
            get_best_scores = []
            if len(good_texts_test3) >= 1:
                for txt in good_texts_test3:
                    good_text_indices = initial_text.all_words_diff(txt)
                    good_text_indices = list(good_text_indices)
                    good_text_indices.sort(key=None, reverse=False)
                    long_good_text_indices = len(good_text_indices)
                    if 1 <= long_good_text_indices <= 10:
                        print("1 <= long_good_text_indices <=10 :")
                        get_good_text = self._reduce_perturbation(initial_text, final_text)

                        if len([get_good_text]) != 0:
                            get_best_texts.append(get_good_text)

                    if 11 <= long_good_text_indices <= 14:
                        print("11 <= long_good_text_indices <= 14")

                        get_good_text = self._reduction_search(initial_result, final_text)
                        good_text_indice = initial_text.all_words_diff(get_good_text)
                        good_text_indice = list(good_text_indice)
                        long_good_text_indice = len(good_text_indice)
                        if 1 <= long_good_text_indice <= 10:
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)
                        else:
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)

                    if 15 <= long_good_text_indices <= 20:
                        print("11 <= long_good_text_indices <= 20")

                        get_good_text = self._reduction_search(initial_result, final_text)
                        good_text_indice = initial_text.all_words_diff(get_good_text)
                        good_text_indice = list(good_text_indice)
                        long_good_text_indice = len(good_text_indice)
                        if 1 <= long_good_text_indice <= 10:
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)
                        else:
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)

                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)

                    if 21 <= long_good_text_indices <= 30:
                        print("21 <= long_good_text_indices <= 30")

                        get_good_text = self._reduction_search(initial_result, final_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        good_text_indice = initial_text.all_words_diff(get_good_text)
                        good_text_indice = list(good_text_indice)
                        long_good_text_indice = len(good_text_indice)
                        if 1 <= long_good_text_indice <= 10:
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)
                        else:
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text = self._reduction_search(initial_result, get_good_text)
                            get_good_text = self._reduce_perturbation(initial_text, get_good_text)

                            if len([get_good_text]) != 0:
                                get_best_texts.append(get_good_text)

                    if 31 <= long_good_text_indices <= 40:
                        print("31 <= long_good_text_indices <= 40")
                        get_good_text = self._reduction_search(initial_result, final_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduction_search(initial_result, get_good_text)
                        get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                        if len([get_good_text]) != 0:
                            get_best_texts.append(get_good_text)

                    elif long_good_text_indices >= 41:
                        result = self._direct_optimization(initial_text, initial_result, final_text)
                        get_good_text = self._reduce_perturbation(initial_text, result.attacked_text)

                        if len([get_good_text]) != 0:
                            get_best_texts.append(get_good_text)

            good_text_indices = initial_text.all_words_diff(final_text)
            good_text_indices = list(good_text_indices)
            good_text_indices.sort(key=None, reverse=False)
            long_good_text_indices = len(good_text_indices)
            if 1 <= long_good_text_indices <= 10:
                print("1 <= long_good_text_indices <=10 :")
                get_good_text = self._reduce_perturbation(initial_text, final_text)

                if len([get_good_text]) != 0:
                    get_best_texts.append(get_good_text)

            if 11 <= long_good_text_indices <= 14:
                print("11 <= long_good_text_indices <= 14")

                get_good_text = self._reduction_search(initial_result, final_text)
                good_text_indice = initial_text.all_words_diff(get_good_text)
                good_text_indice = list(good_text_indice)
                long_good_text_indice = len(good_text_indice)
                if 1 <= long_good_text_indice <= 10:
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)
                else:
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)

                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)

            if 14 <= long_good_text_indices <= 20:
                print("14 <= long_good_text_indices <= 20")

                get_good_text = self._reduction_search(initial_result, final_text)
                good_text_indice = initial_text.all_words_diff(get_good_text)
                good_text_indice = list(good_text_indice)
                long_good_text_indice = len(good_text_indice)
                if 1 <= long_good_text_indice <= 10:
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)
                else:
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)

                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)

            if 21 <= long_good_text_indices <= 30:
                print("21 <= long_good_text_indices <= 30")
                get_good_text = self._reduction_search(initial_result, final_text)
                get_good_text = self._reduction_search(initial_result, get_good_text)
                good_text_indice = initial_text.all_words_diff(get_good_text)
                good_text_indice = list(good_text_indice)
                long_good_text_indice = len(good_text_indice)
                if 1 <= long_good_text_indice <= 10:
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)
                else:
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduction_search(initial_result, get_good_text)
                    get_good_text = self._reduce_perturbation(initial_text, get_good_text)
                    if len([get_good_text]) != 0:
                        get_best_texts.append(get_good_text)

            if 31 <= long_good_text_indices <= 40:

                print("31 <= long_good_text_indices <= 40")

                get_good_text = self._reduction_search(initial_result, final_text)
                get_good_text = self._reduction_search(initial_result, get_good_text)
                get_good_text = self._reduction_search(initial_result, get_good_text)
                get_good_text = self._reduction_search(initial_result, get_good_text)
                get_good_text = self._reduce_perturbation(initial_text, get_good_text)

                if len([get_good_text]) != 0:
                    get_best_texts.append(get_good_text)

            elif long_good_text_indices >= 41:

                result = self._direct_optimization(initial_text, initial_result, final_text)
                get_good_text = self._reduce_perturbation(initial_text, result.attacked_text)

                if len([get_good_text]) != 0:
                    get_best_texts.append(get_good_text)

            if len(get_best_texts) >= 1:
                for txt in get_best_texts:
                    txt_similarity_score = self._get_similarity_score(
                        [txt.text],
                        initial_text.text).tolist()
                    get_best_scores.append(txt_similarity_score)
                    imformation = list(zip(get_best_scores, get_best_texts))
                    imformation.sort(key=lambda score: score[0], reverse=True)
                get_text = imformation[0][1]
                best_result, search_over = self.get_goal_results([get_text])
                return best_result[0]
            else:
                best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
                return best_result[0]

    def _perform_search(self, initial_result):  # 操作搜索
        # Random Initialization 随机初始化，初始化对抗样本
        initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)

        if initial_result.attacked_text.words == initial_adv_text.words:
            return initial_result
        # Search space reduction
        new_text = self._search_space_reduction(initial_result, initial_adv_text)
        results, search_over = self.get_goal_results([new_text])
        if search_over:
            return init_adv_result

        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
            return results[0]

        initial_text = initial_result.attacked_text
        changed_indices = initial_text.all_words_diff(new_text)
        changed_indices = list(changed_indices)
        a = initial_text.text
        print(a)
        # if a == "Following up the 1970s classic horror film Carrie with this offering, is like Ford following the " \
        #         "Mustang with the Edsel. This film was horrendous in every detail. It would have been titled Beverly " \
        #         "Hill 90210 meets Mystery Science Theater 3000, but both of those shows far exceed this tripe. This " \
        #         "film was scarcely a horror film. I timed about 3 minutes of gore and 90 minutes of lame high school " \
        #         "hazing and ritual. Wow, what a surprise, Carrie's weird friend commits suicide! Wow, " \
        #         "Carrie misconstrues her love interests affections! Wow, the in-crowd sets up Carrie! Wow, " \
        #         "the jocks have a sexual scoring contest! What this film needed was way more action and far less " \
        #         "tired teen cliches. This film is totally unviewable.":
        #     getresult, search_over = self.get_goal_results([initial_text])
        #     return getresult[0]

        # Initialize Population 初始化种群
        # population_add = self._initialize_population(
        #     new_text, initial_text, changed_indices
        # )
        get_changed_indice = initial_text.all_words_diff(new_text)
        get_changed_indice = list(get_changed_indice)
        get_changed_indice_length = len(get_changed_indice)
        population = defaultdict
        if get_changed_indice_length <= 20:
            # Initialize Population DE 初始化种群DE
            population = self._initialize_populationDE(
                new_text, initial_text, changed_indices
            )
        elif get_changed_indice_length <= 30:
            new_text = self._reduction_search(initial_result, new_text)
            population = self._initialize_populationDE(
                new_text, initial_text, changed_indices
            )

        elif get_changed_indice_length <= 40:
            new_text = self._reduction_search(initial_result, new_text)
            new_text = self._reduction_search(initial_result, new_text)
            population = self._initialize_populationDE(
                new_text, initial_text, changed_indices
            )
        else:
            new_text = self._reduction_search(initial_result, new_text)
            new_text = self._reduction_search(initial_result, new_text)
            new_text = self._reduction_search(initial_result, new_text)
            population = self._initialize_populationDE(
                new_text, initial_text, changed_indices
            )
        results, search_over = self.get_goal_results([new_text])

        if len(population) == 0:
            print("level-0")
            results0 = self._level_0_optimization(initial_text, initial_result, initial_adv_text)
            return results0

        if len(changed_indices) == 1:
            print("level-1")
            results1 = self._level_1_optimization(population, initial_result)
            return results1

        if len(changed_indices) == 2:
            print("level-2")
            results2 = self._level_2_optimization(changed_indices, population, initial_text, initial_result, results)
            return results2

        if len(changed_indices) >= 3:
            print("level-3")
            results3 = self._level_3_optimization(initial_text, initial_result, population, new_text, results
                                                  )
            txt_similarity_score = self._get_similarity_score(
                [results3.attacked_text.text],
                initial_text.text).tolist()
            txt_similarity_score = txt_similarity_score[0]
            print("level-3 get score", txt_similarity_score)
            return results3

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "max_replacements_per_index"]

        # if len(population) == 0:
        #     print("level-0")
        #     # result = self._optimize_words(new_text, initial_text, changed_indices)
        #     # 只需要，对这边的对抗样本中，的位置进行同义词替换，找到效果最好的句子。将他们保存
        #     text_indices = initial_text.all_words_diff(results[0].attacked_text)
        #     text_indices = list(text_indices)
        #     text_indices.sort(key=None, reverse=False)
        #     long_text_indices = len(text_indices)
        #     if 1 <= long_text_indices <= 10:
        #         print("1 <= long_text_indices <= 11")
        #         get_good_text = self._reduce_perturbation(initial_text, results[0].attacked_text)
        #         get_good_text_result, search_over = self.get_goal_results([get_good_text])
        #         return get_good_text_result[0]  # 搜索空间减少后的句子
        #     elif long_text_indices >= 11:
        #         get_good_text = self._reduction_search(initial_result, results[0].attacked_text)
        #         get_good_text_results, search_over = self.get_goal_results([get_good_text])
        #         if search_over:
        #             return results[0]
        #         if get_good_text_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #             text_indices = initial_text.all_words_diff(get_good_text)
        #             text_indices = list(text_indices)
        #             text_indices.sort(key=None, reverse=False)
        #             long_text_indices = len(text_indices)
        #             if long_text_indices <= 10:
        #                 get_best_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_best_text_result, search_over = self.get_goal_results([get_best_text])
        #                 return get_best_text_result[0]
        #             if long_text_indices >= 11:
        #                 get_good_text = self._reduction_search(initial_result, get_good_text)
        #                 get_good_text_results, search_over = self.get_goal_results([get_good_text])
        #                 if search_over:
        #                     return results[0]
        #                 if get_good_text_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                     text_indices = initial_text.all_words_diff(get_good_text)
        #                     text_indices = list(text_indices)
        #                     text_indices.sort(key=None, reverse=False)
        #                     long_text_indices = len(text_indices)
        #                     if long_text_indices <= 10:
        #                         get_best_text = self._reduce_perturbation(initial_text, get_good_text)
        #                         get_best_text_result, search_over = self.get_goal_results([get_best_text])
        #                         return get_best_text_result[0]
        #                     else:
        #                         return get_good_text_results[0]
        #
        # if len(changed_indices) == 1:
        #     print("level-1")
        #     similarity_score = []
        #     pop_members = []
        #     best_result = defaultdict()
        #     for pop_member in population:
        #         similarity_score.append(pop_member.attributes["similarity_score"])
        #         pop_members.append(pop_member.attacked_text)
        #         imformation = list(zip(similarity_score, pop_members))
        #         imformation.sort(key=lambda score: score[0], reverse=True)
        #         final_text = imformation[0][1]
        #         final_similarity_score = imformation[0][0]
        #         result_pop_member = PopulationMember(
        #             final_text,
        #             attributes={
        #                 "similarity_score": final_similarity_score,
        #                 "valid_adversarial_example": True,
        #             },
        #         )
        #         print("level-1 get score", final_similarity_score)
        #         if final_similarity_score >= 0.68:
        #             best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #             return best_result[0]
        #         initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)
        #         if initial_result.attacked_text.words == initial_adv_text.words:
        #             return initial_result
        #         new_text = self._search_space_reduction(initial_result, initial_adv_text)
        #         results, search_over = self.get_goal_results([new_text])
        #         if search_over:
        #             return init_adv_result
        #         if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
        #             return results[0]
        #         initial_text = initial_result.attacked_text
        #         changed_indices = initial_text.all_words_diff(new_text)
        #         changed_indices = list(changed_indices)
        #         self._search_over = False
        #         population = self._initialize_populationDE(
        #             new_text, initial_text, changed_indices
        #         )
        #
        #         if len(population) == 0:
        #             print("level-0")
        #             return results[0]
        #         if len(changed_indices) == 1:
        #             print("level-1")
        #             similarity_score = []
        #             pop_members = []
        #             for pop_member in population:
        #                 similarity_score.append(pop_member.attributes["similarity_score"])
        #                 pop_members.append(pop_member.attacked_text)
        #                 imformation = list(zip(similarity_score, pop_members))
        #                 imformation.sort(key=lambda score: score[0], reverse=True)
        #                 final_text = imformation[0][1]
        #                 final_similarity_score = imformation[0][0]
        #                 result_pop_member = PopulationMember(
        #                     final_text,
        #                     attributes={
        #                         "similarity_score": final_similarity_score,
        #                         "valid_adversarial_example": True,
        #                     },
        #                 )
        #                 print("level-1 ? get", final_similarity_score)
        #                 if final_similarity_score >= 0.68:
        #                     best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                     return best_result[0]
        #                 else:
        #                     initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(
        #                         initial_result)
        #                     if initial_result.attacked_text.words == initial_adv_text.words:
        #                         return initial_result
        #                     new_text = self._search_space_reduction(initial_result, initial_adv_text)
        #                     results, search_over = self.get_goal_results([new_text])
        #                     if search_over:
        #                         return init_adv_result
        #                     if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
        #                         return results[0]
        #                     initial_text = initial_result.attacked_text
        #                     changed_indices = initial_text.all_words_diff(new_text)
        #                     changed_indices = list(changed_indices)
        #                     self._search_over = False
        #                     population = self._initialize_populationDE(
        #                         new_text, initial_text, changed_indices
        #                     )
        #
        #                     if len(population) == 0:
        #                         print("level-0")
        #                         return results[0]
        #                     if len(changed_indices) == 1:
        #                         print("level-1")
        #                         similarity_score = []
        #                         pop_members = []
        #                         for pop_member in population:
        #                             similarity_score.append(pop_member.attributes["similarity_score"])
        #                             pop_members.append(pop_member.attacked_text)
        #                             imformation = list(zip(similarity_score, pop_members))
        #                             imformation.sort(key=lambda score: score[0], reverse=True)
        #                             final_text = imformation[0][1]
        #                             final_similarity_score = imformation[0][0]
        #                             result_pop_member = PopulationMember(
        #                                 final_text,
        #                                 attributes={
        #                                     "similarity_score": final_similarity_score,
        #                                     "valid_adversarial_example": True,
        #                                 },
        #                             )
        #                             print("level-1 ?? get score", final_similarity_score)
        #                             best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                             return best_result[0]
        #                     if len(changed_indices) == 2:
        #                         print("level-2")
        #
        #                         perturbed_text_text1 = []
        #                         perturbed_text_text2 = []
        #                         get_index1 = []
        #                         get_index2 = []
        #                         index1_to_word = []
        #                         index2_to_word = []
        #                         all_index = changed_indices
        #                         get_one_index = random.sample(changed_indices, 1)
        #                         for i in get_one_index:
        #                             index_1 = i
        #                             all_index.remove(i)
        #                         for j in all_index:
        #                             index_2 = j
        #                         for pop_members in population:
        #                             if pop_members.attributes["index"] == index_1:
        #                                 get_index1.append(index_1)
        #                                 index1_to_word.append(pop_members.attacked_text.words[index_1])
        #                             else:
        #                                 get_index2.append(index_2)
        #                                 index2_to_word.append(pop_members.attacked_text.words[index_2])
        #
        #                         for pop_members in population:
        #                             if pop_members.attributes["index"] == index_1:
        #                                 new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                     get_index2, index2_to_word
        #                                 )
        #                                 perturbed_text_text1.append(new_text)
        #                                 # print("perturbed_text_text1", perturbed_text_text1)
        #                             if pop_members.attributes["index"] == index_2:
        #                                 new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                     get_index1, index1_to_word
        #                                 )
        #                                 perturbed_text_text2.append(new_text)
        #                                 # print("perturbed_text_text2", perturbed_text_text2)
        #                         select_text = perturbed_text_text1 + perturbed_text_text2
        #                         text_results, search_over = self.get_goal_results(select_text)
        #                         select_texts = []
        #                         for x in range(len(text_results)):
        #                             if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                                 select_texts.append(text_results[x].attacked_text)
        #                         if len(select_texts) == 0:
        #                             return results[0]
        #                         txt_similarity_scores = []
        #                         for txt in select_texts:
        #                             txt_similarity_score = self._get_similarity_score(
        #                                 [txt.text],
        #                                 initial_text.text).tolist()
        #                             txt_similarity_scores.append(txt_similarity_score)
        #                         imformation = list(zip(txt_similarity_scores, select_texts))
        #                         imformation.sort(key=lambda score: score[0], reverse=True)
        #                         get_text = imformation[0][1]
        #                         get_similarity_score = float((imformation[0][0])[0])
        #                         get_indices = list(initial_text.all_words_diff(get_text))
        #                         final_texts = []
        #                         final_scores = []
        #                         for idx in get_indices:
        #                             final_text = get_text.replace_word_at_index(
        #                                 idx, initial_text.words[idx]
        #                             )
        #                             get_results, search_over = self.get_goal_results([final_text])
        #
        #                             if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                                 similarity_score = self._get_similarity_score(
        #                                     [final_text.text], initial_text.text
        #                                 ).tolist()[0]
        #                                 final_texts.append(final_text)
        #                                 final_scores.append(similarity_score)
        #                                 # print("final_texts", final_texts, "final_scores", final_scores)
        #                         if len(final_texts) >= 1:
        #                             final_information = list(zip(final_scores, final_texts))
        #                             final_information.sort(key=lambda score: score[0], reverse=True)
        #                             get_final_text = final_information[0][1]
        #                             get_final_score = float((final_information[0][0]))
        #                             result_pop_member = PopulationMember(
        #                                 get_final_text,
        #                                 attributes={
        #                                     "similarity_score": get_final_score,
        #                                     "valid_adversarial_example": True,
        #                                 },
        #                             )
        #                             print("level-2 ? get score", get_final_score)
        #                             best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                             return best_result[0]
        #                         else:
        #                             similarity_score = []
        #                             pop_members = []
        #                             for pop_member in population:
        #                                 similarity_score.append(pop_member.attributes["similarity_score"])
        #                                 pop_members.append(pop_member.attacked_text)
        #                                 imformation = list(zip(similarity_score, pop_members))
        #                                 imformation.sort(key=lambda score: score[0], reverse=True)
        #                                 final_text = imformation[0][1]
        #                                 final_score = imformation[0][0]
        #                                 result_pop_member = PopulationMember(
        #                                     final_text,
        #                                     attributes={
        #                                         "similarity_score": final_score,
        #                                         "valid_adversarial_example": True,
        #                                     },
        #                                 )
        #                                 print("level-2 ?? get score", final_score)
        #                                 if final_score >= 0.70:
        #                                     best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                                     return best_result[0]
        #                                 else:
        #                                     initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(
        #                                         initial_result)
        #                                     if initial_result.attacked_text.words == initial_adv_text.words:
        #                                         return initial_result
        #                                     new_text = self._search_space_reduction(initial_result, initial_adv_text)
        #                                     results, search_over = self.get_goal_results([new_text])
        #                                     if search_over:
        #                                         return init_adv_result
        #                                     if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
        #                                         return results[0]
        #                                     initial_text = initial_result.attacked_text
        #                                     changed_indices = initial_text.all_words_diff(new_text)
        #                                     changed_indices = list(changed_indices)
        #                                     self._search_over = False
        #                                     population = self._initialize_populationDE(
        #                                         new_text, initial_text, changed_indices
        #                                     )
        #
        #                                     if len(population) == 0:
        #                                         print("level-0")
        #                                         return results[0]
        #                                     if len(changed_indices) == 1:
        #                                         print("level-1")
        #                                         similarity_score = []
        #                                         pop_members = []
        #                                         for pop_member in population:
        #                                             similarity_score.append(pop_member.attributes["similarity_score"])
        #                                             pop_members.append(pop_member.attacked_text)
        #                                             imformation = list(zip(similarity_score, pop_members))
        #                                             imformation.sort(key=lambda score: score[0], reverse=True)
        #                                             final_text = imformation[0][1]
        #                                             final_similarity_score = imformation[0][0]
        #                                             result_pop_member = PopulationMember(
        #                                                 final_text,
        #                                                 attributes={
        #                                                     "similarity_score": final_similarity_score,
        #                                                     "valid_adversarial_example": True,
        #                                                 },
        #                                             )
        #                                             print("level-1 ??? get score", final_similarity_score)
        #                                             best_result, _ = self.get_goal_results(
        #                                                 [result_pop_member.attacked_text])
        #                                             return best_result[0]
        #                                     if len(changed_indices) == 2:
        #                                         print("level-2")
        #                                         perturbed_text_text1 = []
        #                                         perturbed_text_text2 = []
        #                                         get_index1 = []
        #                                         get_index2 = []
        #                                         index1_to_word = []
        #                                         index2_to_word = []
        #                                         all_index = changed_indices
        #                                         get_one_index = random.sample(changed_indices, 1)
        #                                         for i in get_one_index:
        #                                             index_1 = i
        #                                             all_index.remove(i)
        #                                         for j in all_index:
        #                                             index_2 = j
        #                                         for pop_members in population:
        #                                             if pop_members.attributes["index"] == index_1:
        #                                                 get_index1.append(index_1)
        #                                                 index1_to_word.append(pop_members.attacked_text.words[index_1])
        #                                             else:
        #                                                 get_index2.append(index_2)
        #                                                 index2_to_word.append(pop_members.attacked_text.words[index_2])
        #                                         for pop_members in population:
        #                                             if pop_members.attributes["index"] == index_1:
        #                                                 new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                                     get_index2, index2_to_word
        #                                                 )
        #                                                 perturbed_text_text1.append(new_text)
        #
        #                                             if pop_members.attributes["index"] == index_2:
        #                                                 new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                                     get_index1, index1_to_word
        #                                                 )
        #                                                 perturbed_text_text2.append(new_text)
        #                                         select_text = perturbed_text_text1 + perturbed_text_text2
        #                                         text_results, search_over = self.get_goal_results(select_text)
        #                                         select_texts = []
        #                                         for x in range(len(text_results)):
        #                                             if text_results[
        #                                                 x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                                                 select_texts.append(text_results[x].attacked_text)
        #                                         if len(select_texts) == 0:
        #                                             return results[0]
        #                                         txt_similarity_scores = []
        #                                         for txt in select_texts:
        #                                             txt_similarity_score = self._get_similarity_score(
        #                                                 [txt.text],
        #                                                 initial_text.text).tolist()
        #                                             txt_similarity_scores.append(txt_similarity_score)
        #                                         imformation = list(zip(txt_similarity_scores, select_texts))
        #                                         imformation.sort(key=lambda score: score[0], reverse=True)
        #                                         get_text = imformation[0][1]
        #                                         get_indices = list(initial_text.all_words_diff(get_text))
        #                                         final_texts = []
        #                                         final_scores = []
        #                                         for idx in get_indices:
        #                                             final_text = get_text.replace_word_at_index(
        #                                                 idx, initial_text.words[idx]
        #                                             )
        #                                             get_results, search_over = self.get_goal_results([final_text])
        #                                             if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                                                 similarity_score = self._get_similarity_score(
        #                                                     [final_text.text], initial_text.text
        #                                                 ).tolist()[0]
        #                                                 final_texts.append(final_text)
        #                                                 final_scores.append(similarity_score)
        #                                         if len(final_texts) >= 1:
        #                                             final_information = list(zip(final_scores, final_texts))
        #                                             final_information.sort(key=lambda score: score[0], reverse=True)
        #                                             get_final_text = final_information[0][1]
        #                                             get_final_score = float((final_information[0][0]))
        #                                             result_pop_member = PopulationMember(
        #                                                 get_final_text,
        #                                                 attributes={
        #                                                     "similarity_score": get_final_score,
        #                                                     "valid_adversarial_example": True,
        #                                                 },
        #                                             )
        #                                             print("levle-2????++ get score", get_final_score)
        #                                             best_result, _ = self.get_goal_results(
        #                                                 [result_pop_member.attacked_text])
        #                                             return best_result[0]
        #                                         else:
        #                                             similarity_score = []
        #                                             pop_members = []
        #                                             for pop_member in population:
        #                                                 similarity_score.append(
        #                                                     pop_member.attributes["similarity_score"])
        #                                                 pop_members.append(pop_member.attacked_text)
        #                                                 imformation = list(zip(similarity_score, pop_members))
        #                                                 imformation.sort(key=lambda score: score[0], reverse=True)
        #                                                 final_text = imformation[0][1]
        #                                                 final_score = imformation[0][0]
        #                                                 result_pop_member = PopulationMember(
        #                                                     final_text,
        #                                                     attributes={
        #                                                         "similarity_score": final_score,
        #                                                         "valid_adversarial_example": True,
        #                                                     },
        #                                                 )
        #                                                 print("level-2 ?????+ get score", final_score)
        #                                                 best_result, _ = self.get_goal_results(
        #                                                     [result_pop_member.attacked_text])
        #                                                 return best_result[0]
        #                                     if len(changed_indices) >= 3:  # 这边还需要替换成最终的结果
        #                                         print("level-3")
        #                                         similarity_score = []
        #                                         pop_members = []
        #                                         for pop_member in population:
        #                                             similarity_score.append(pop_member.attributes["similarity_score"])
        #                                             pop_members.append(pop_member.attacked_text)
        #                                             imformation = list(zip(similarity_score, pop_members))
        #                                             imformation.sort(key=lambda score: score[0], reverse=True)
        #                                             final_text = imformation[0][1]
        #                                             final_score = imformation[0][0]
        #                                             result_pop_member = PopulationMember(
        #                                                 final_text,
        #                                                 attributes={
        #                                                     "similarity_score": final_score,
        #                                                     "valid_adversarial_example": True,
        #                                                 },
        #                                             )
        #                                             print("mid_score", final_score)
        #                                             best_result, _ = self.get_goal_results(
        #                                                 [result_pop_member.attacked_text])
        #                                             return best_result[0]
        #
        # if len(changed_indices) == 2:
        #     print("level-2")
        #
        #     perturbed_text_text1 = []
        #     perturbed_text_text2 = []
        #     get_index1 = []
        #     get_index2 = []
        #     index1_to_word = []
        #     index2_to_word = []
        #     all_index = changed_indices
        #     get_one_index = random.sample(changed_indices, 1)
        #     for i in get_one_index:
        #         index_1 = i
        #         all_index.remove(i)
        #     for j in all_index:
        #         index_2 = j
        #     for pop_members in population:
        #         if pop_members.attributes["index"] == index_1:
        #             get_index1.append(index_1)
        #             index1_to_word.append(pop_members.attacked_text.words[index_1])
        #         else:
        #             get_index2.append(index_2)
        #             index2_to_word.append(pop_members.attacked_text.words[index_2])
        #
        #     for pop_members in population:
        #         if pop_members.attributes["index"] == index_1:
        #             new_text = pop_members.attacked_text.replace_words_at_indices(
        #                 get_index2, index2_to_word
        #             )
        #             perturbed_text_text1.append(new_text)
        #             # print("perturbed_text_text1", perturbed_text_text1)
        #         if pop_members.attributes["index"] == index_2:
        #             new_text = pop_members.attacked_text.replace_words_at_indices(
        #                 get_index1, index1_to_word
        #             )
        #             perturbed_text_text2.append(new_text)
        #             # print("perturbed_text_text2", perturbed_text_text2)
        #     select_text = perturbed_text_text1 + perturbed_text_text2
        #     text_results, search_over = self.get_goal_results(select_text)
        #     select_texts = []
        #     for x in range(len(text_results)):
        #         if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #             select_texts.append(text_results[x].attacked_text)
        #     if len(select_texts) == 0:
        #         return results[0]
        #     txt_similarity_scores = []
        #     for txt in select_texts:
        #         txt_similarity_score = self._get_similarity_score(
        #             [txt.text],
        #             initial_text.text).tolist()
        #         txt_similarity_scores.append(txt_similarity_score)
        #     imformation = list(zip(txt_similarity_scores, select_texts))
        #     imformation.sort(key=lambda score: score[0], reverse=True)
        #     get_text = imformation[0][1]
        #     get_similarity_score = float((imformation[0][0])[0])
        #     get_indices = list(initial_text.all_words_diff(get_text))
        #     final_texts = []
        #     final_scores = []
        #     for idx in get_indices:
        #         final_text = get_text.replace_word_at_index(
        #             idx, initial_text.words[idx]
        #         )
        #         get_results, search_over = self.get_goal_results([final_text])
        #
        #         if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #             similarity_score = self._get_similarity_score(
        #                 [final_text.text], initial_text.text
        #             ).tolist()[0]
        #             final_texts.append(final_text)
        #             final_scores.append(similarity_score)
        #             # print("final_texts", final_texts, "final_scores", final_scores)
        #     if len(final_texts) >= 1:
        #         final_information = list(zip(final_scores, final_texts))
        #         final_information.sort(key=lambda score: score[0], reverse=True)
        #         get_final_text = final_information[0][1]
        #         get_final_score = float((final_information[0][0]))
        #         result_pop_member = PopulationMember(
        #             get_final_text,
        #             attributes={
        #                 "similarity_score": get_final_score,
        #                 "valid_adversarial_example": True,
        #             },
        #         )
        #         print("levle-2 get ?+ final score", get_final_score)
        #         best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #         return best_result[0]
        #     else:
        #         similarity_score = []
        #         pop_members = []
        #         for pop_member in population:
        #             similarity_score.append(pop_member.attributes["similarity_score"])
        #             pop_members.append(pop_member.attacked_text)
        #             imformation = list(zip(similarity_score, pop_members))
        #             imformation.sort(key=lambda score: score[0], reverse=True)
        #             final_text = imformation[0][1]
        #             final_score = imformation[0][0]
        #             result_pop_member = PopulationMember(
        #                 final_text,
        #                 attributes={
        #                     "similarity_score": final_score,
        #                     "valid_adversarial_example": True,
        #                 },
        #             )
        #             print("level-2 +++? get score", final_score)
        #             if final_score >= 0.70:
        #                 best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                 return best_result[0]
        #             else:
        #                 initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)
        #                 if initial_result.attacked_text.words == initial_adv_text.words:
        #                     return initial_result
        #                 new_text = self._search_space_reduction(initial_result, initial_adv_text)
        #                 results, search_over = self.get_goal_results([new_text])
        #                 if search_over:
        #                     return init_adv_result
        #                 if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
        #                     return results[0]
        #                 initial_text = initial_result.attacked_text
        #                 changed_indices = initial_text.all_words_diff(new_text)
        #                 changed_indices = list(changed_indices)
        #                 self._search_over = False
        #                 population = self._initialize_populationDE(
        #                     new_text, initial_text, changed_indices
        #                 )
        #
        #                 if len(population) == 0:
        #                     print("level-0")
        #                     return results[0]
        #                 if len(changed_indices) == 1:
        #                     print("level-1")
        #                     similarity_score = []
        #                     pop_members = []
        #                     for pop_member in population:
        #                         similarity_score.append(pop_member.attributes["similarity_score"])
        #                         pop_members.append(pop_member.attacked_text)
        #                         imformation = list(zip(similarity_score, pop_members))
        #                         imformation.sort(key=lambda score: score[0], reverse=True)
        #                         final_text = imformation[0][1]
        #                         final_similarity_score = imformation[0][0]
        #                         result_pop_member = PopulationMember(
        #                             final_text,
        #                             attributes={
        #                                 "similarity_score": final_similarity_score,
        #                                 "valid_adversarial_example": True,
        #                             },
        #                         )
        #                         print("level-1 !++ get score", final_similarity_score)
        #                         best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                         return best_result[0]
        #                 if len(changed_indices) == 2:
        #                     print("level-2")
        #                     perturbed_text_text1 = []
        #                     perturbed_text_text2 = []
        #                     get_index1 = []
        #                     get_index2 = []
        #                     index1_to_word = []
        #                     index2_to_word = []
        #                     all_index = changed_indices
        #                     get_one_index = random.sample(changed_indices, 1)
        #                     for i in get_one_index:
        #                         index_1 = i
        #                         all_index.remove(i)
        #                     for j in all_index:
        #                         index_2 = j
        #                     for pop_members in population:
        #                         if pop_members.attributes["index"] == index_1:
        #                             get_index1.append(index_1)
        #                             index1_to_word.append(pop_members.attacked_text.words[index_1])
        #                         else:
        #                             get_index2.append(index_2)
        #                             index2_to_word.append(pop_members.attacked_text.words[index_2])
        #                     for pop_members in population:
        #                         if pop_members.attributes["index"] == index_1:
        #                             new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                 get_index2, index2_to_word
        #                             )
        #                             perturbed_text_text1.append(new_text)
        #
        #                         if pop_members.attributes["index"] == index_2:
        #                             new_text = pop_members.attacked_text.replace_words_at_indices(
        #                                 get_index1, index1_to_word
        #                             )
        #                             perturbed_text_text2.append(new_text)
        #                     select_text = perturbed_text_text1 + perturbed_text_text2
        #                     text_results, search_over = self.get_goal_results(select_text)
        #                     select_texts = []
        #                     for x in range(len(text_results)):
        #                         if text_results[x].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                             select_texts.append(text_results[x].attacked_text)
        #                     if len(select_texts) == 0:
        #                         return results[0]
        #                     txt_similarity_scores = []
        #                     for txt in select_texts:
        #                         txt_similarity_score = self._get_similarity_score(
        #                             [txt.text],
        #                             initial_text.text).tolist()
        #                         txt_similarity_scores.append(txt_similarity_score)
        #                     imformation = list(zip(txt_similarity_scores, select_texts))
        #                     imformation.sort(key=lambda score: score[0], reverse=True)
        #                     get_text = imformation[0][1]
        #                     get_indices = list(initial_text.all_words_diff(get_text))
        #                     final_texts = []
        #                     final_scores = []
        #                     for idx in get_indices:
        #                         final_text = get_text.replace_word_at_index(
        #                             idx, initial_text.words[idx]
        #                         )
        #                         get_results, search_over = self.get_goal_results([final_text])
        #                         if get_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                             similarity_score = self._get_similarity_score(
        #                                 [final_text.text], initial_text.text
        #                             ).tolist()[0]
        #                             final_texts.append(final_text)
        #                             final_scores.append(similarity_score)
        #                     if len(final_texts) >= 1:
        #                         final_information = list(zip(final_scores, final_texts))
        #                         final_information.sort(key=lambda score: score[0], reverse=True)
        #                         get_final_text = final_information[0][1]
        #                         get_final_score = float((final_information[0][0]))
        #                         result_pop_member = PopulationMember(
        #                             get_final_text,
        #                             attributes={
        #                                 "similarity_score": get_final_score,
        #                                 "valid_adversarial_example": True,
        #                             },
        #                         )
        #                         print("level-2 ?+++-+ get score", get_final_score)
        #                         best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                         return best_result[0]
        #                     else:
        #                         similarity_score = []
        #                         pop_members = []
        #                         for pop_member in population:
        #                             similarity_score.append(pop_member.attributes["similarity_score"])
        #                             pop_members.append(pop_member.attacked_text)
        #                             imformation = list(zip(similarity_score, pop_members))
        #                             imformation.sort(key=lambda score: score[0], reverse=True)
        #                             final_text = imformation[0][1]
        #                             final_score = imformation[0][0]
        #                             result_pop_member = PopulationMember(
        #                                 final_text,
        #                                 attributes={
        #                                     "similarity_score": final_score,
        #                                     "valid_adversarial_example": True,
        #                                 },
        #                             )
        #                             print("level-2 !+???+ get score", final_score)
        #                             best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                             return best_result[0]
        #                 if len(changed_indices) >= 3:  # 这边还需要替换成最终的结果
        #                     print("level-3")
        #                     similarity_score = []
        #                     pop_members = []
        #                     for pop_member in population:
        #                         similarity_score.append(pop_member.attributes["similarity_score"])
        #                         pop_members.append(pop_member.attacked_text)
        #                         imformation = list(zip(similarity_score, pop_members))
        #                         imformation.sort(key=lambda score: score[0], reverse=True)
        #                         final_text = imformation[0][1]
        #                         final_score = imformation[0][0]
        #                         result_pop_member = PopulationMember(
        #                             final_text,
        #                             attributes={
        #                                 "similarity_score": final_score,
        #                                 "valid_adversarial_example": True,
        #                             },
        #                         )
        #                         print("mid_score+", final_score)
        #                         best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                         return best_result[0]

        # if len(changed_indices) >= 3:
        #
        #     index_to_replace = []
        #     words_to_replace = []
        #     scores_to_replace = []
        #     changed_indices = initial_text.all_words_diff(new_text)
        #     changed_indices = list(changed_indices)
        #     # print("??changed_indices", changed_indices)
        #     for idx in changed_indices:
        #         for pop_members in population:
        #             if pop_members.attributes["index"] == idx:
        #
        #                 index_to_replace.append(idx)
        #                 words_to_replace.append(pop_members.attacked_text.words[idx])
        #                 if pop_members.attributes["index"] == idx \
        #                         and pop_members.attributes["index_to_word"] == pop_members.attacked_text.words[idx]:
        #                     scores_to_replace.append(pop_members.attributes["similarity_score"])
        #     index_to_replace.reverse()
        #     words_to_replace.reverse()
        #     scores_to_replace.reverse()
        #     # print("index_to_replace", index_to_replace, "words_to_replace", words_to_replace, "scores_to_replace",
        #     #       scores_to_replace)
        #
        #     similarity_scores = []
        #     pop_members = []
        #     good_pop_member = defaultdict
        #     good_pop_member_word = defaultdict
        #     for pop_member in population:
        #         similarity_scores.append(pop_member.attributes["similarity_score"])
        #         pop_members.append(pop_member.attacked_text)
        #         imformation = list(zip(similarity_scores, pop_members))
        #         imformation.sort(key=lambda score: score[0], reverse=True)
        #         final_text = imformation[0][1]
        #         final_score = imformation[0][0]
        #         good_pop_member = PopulationMember(
        #             final_text,
        #             attributes={
        #                 "similarity_score": final_score,
        #                 "valid_adversarial_example": True,
        #             }
        #         )
        #         # 选择语义相似度最高的种群，优化,将可以替换的位置，都替换成语义最高的单词
        #     mid_text = good_pop_member.attacked_text.replace_words_at_indices(
        #         index_to_replace, words_to_replace
        #     )
        #     mid_text_result, search_over = self.get_goal_results([mid_text])
        #     if mid_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #         print("ok,go on")
        #         good_pop_member_indice = results[0].attacked_text.all_words_diff(good_pop_member.attacked_text)
        #         good_pop_member_indice = list(good_pop_member_indice)  # 调整原始位置的单词，此处索引全部为“1”
        #         if len(good_pop_member_indice) >= 1:
        #             good_text = defaultdict  # good_text是目前，将扰动位置都换成了，最高语义的单词的对抗样本，扰动率高
        #             for i in good_pop_member_indice:
        #                 good_pop_member_word = good_pop_member.attacked_text.words[i]
        #                 good_text = mid_text.replace_words_at_indices(
        #                     good_pop_member_indice, [good_pop_member_word]
        #                 )
        #             good_text_indices = initial_text.all_words_diff(good_text)
        #             good_text_indices = list(good_text_indices)
        #             good_text_indices.sort(key=None, reverse=False)
        #             long_good_text_indices = len(good_text_indices)
        #             get_good_texts = []
        #             a = initial_text.text
        #
        #             if 1 <= long_good_text_indices <= 10:
        #                 print("1 <= long_good_text_indices <= 10")
        #                 get_good_text = self._reduce_perturbation(initial_text, good_text)
        #                 get_good_texts.append(get_good_text)
        #
        #             if 11 <= long_good_text_indices <= 50:
        #                 print("11 <= long_good_text_indices <= 50")
        #                 get_good_text = self._reduce_perturbation_add(initial_text, good_text)
        #                 get_good_texts.append(get_good_text)
        #             elif 51 <= long_good_text_indices <= 100:
        #                 print("51 <= long_good_text_indices <= 100")
        #                 get_good_text = self._reduce_perturbation(initial_text, good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_texts.append(get_good_text)
        #             elif 101 <= long_good_text_indices <= 150:
        #                 print("101 <= long_good_text_indices <= 150")
        #                 get_good_text = self._reduce_perturbation(initial_text, good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_texts.append(get_good_text)
        #             elif 151 <= long_good_text_indices <= 200:
        #                 get_good_text = self._reduce_perturbation(initial_text, good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_text = self._reduce_perturbation(initial_text, get_good_text)
        #                 get_good_texts.append(get_good_text)
        #
        #             indexs_test = []
        #             words_test = []
        #             for i in good_text_indices:
        #                 test_text3 = good_text.replace_words_at_indices(
        #                     [i], [initial_text.words[i]]
        #                 )
        #                 test_text_result, search_over = self.get_goal_results([test_text3])
        #                 if test_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                     indexs_test.append(i)
        #                     words_test.append(initial_text.words[i])
        #                     test_text4 = good_text.replace_words_at_indices(
        #                         indexs_test, words_test
        #                     )
        #                     test_text_result, search_over = self.get_goal_results([test_text4])
        #                     if test_text_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                         get_good_texts.append(test_text_result[0].attacked_text)
        #             final_good_scores = []
        #             get_final_text = defaultdict
        #             for txt in get_good_texts:
        #                 similarity_score = self._get_similarity_score(
        #                     [txt.text], initial_text.text
        #                 ).tolist()[0]
        #                 final_good_scores.append(similarity_score)
        #                 final_information = list(zip(final_good_scores, get_good_texts))
        #                 final_information.sort(key=lambda score: score[0], reverse=True)
        #                 get_final_text = final_information[0][1]
        #             best_text_results, search_over = self.get_goal_results([get_final_text])
        #             if best_text_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #                 return best_text_results[0]
        #             else:
        #                 best_result, _ = self.get_goal_results([good_text])
        #                 return best_result[0]
        #         else:
        #             best_result, _ = self.get_goal_results([mid_text])
        #             return best_result[0]
        #             # good_text = mid_text
        #             # 将每个索引位置的单词，替换为语义相似度最高的单词
        #             # 降低扰动率
        #
        #     else:
        #         print("no,back")  # 这边的效果不好，需要逐一替换
        #         a = initial_text.text
        #         print(a)
        #
        #         similarity_score = []
        #         pop_members = []
        #         final_text = defaultdict
        #         result_pop_member = defaultdict
        #         for pop_member in population:
        #             similarity_score.append(pop_member.attributes["similarity_score"])
        #             pop_members.append(pop_member.attacked_text)
        #             imformation = list(zip(similarity_score, pop_members))
        #             imformation.sort(key=lambda score: score[0], reverse=True)
        #             final_text = imformation[0][1]
        #             final_score = imformation[0][0]
        #             result_pop_member = PopulationMember(
        #                 final_text,
        #                 attributes={
        #                     "similarity_score": final_score,
        #                     "valid_adversarial_example": True,
        #                 },
        #             )
        #         good_text_indices = initial_text.all_words_diff(final_text)
        #         good_text_indices = list(good_text_indices)
        #         good_text_indices.sort(key=None, reverse=False)
        #         long_good_text_indices = len(good_text_indices)
        #         if 1 <= long_good_text_indices <= 10:
        #             get_good_text = self._reduce_perturbation(initial_text, final_text)
        #             if len([get_good_text]) != 0:
        #                 best_result, _ = self.get_goal_results([get_good_text])
        #                 return best_result[0]
        #             else:
        #                 best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                 return best_result[0]
        #         elif long_good_text_indices >= 11:
        #             get_good_text = self._reduce_perturbation_add(initial_text, final_text)
        #             if len([get_good_text]) != 0:
        #                 best_result, _ = self.get_goal_results([get_good_text])
        #                 return best_result[0]
        #             else:
        #                 best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #                 return best_result[0]
        #         else:
        #             best_result, _ = self.get_goal_results([result_pop_member.attacked_text])
        #             return best_result[0]
