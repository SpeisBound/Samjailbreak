# import torch
# gpu_count = torch.cuda.device_count()
# print(f"Number of available GPUs: {gpu_count}")
# current_gpu_index = torch.cuda.current_device()
# print(f"Current GPU index: {current_gpu_index}")
# gpu_name = torch.cuda.get_device_name(current_gpu_index)
# print(f"GPU name: {gpu_name}")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import collections
import string
import sys
import re
import torch
import textattack
from textattack.attack_args import AttackArgs
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.datasets_text_attack import Dataset
from textattack.goal_functions.classification.untargeted_classification import UntargetedClassification
from utils import FileLogger

sys.setrecursionlimit(8000)
ATTACK_RECIPE_NAMES = {
    "alzantot": "textattack.attack_recipes.GeneticAlgorithmAlzantot2018",
    "bae": "textattack.attack_recipes.BAEGarg2019",
    "bert-attack": "textattack.attack_recipes.BERTAttackLi2020",
    "faster-alzantot": "textattack.attack_recipes.FasterGeneticAlgorithmJia2019",
    "deepwordbug": "textattack.attack_recipes.DeepWordBugGao2018",
    "hotflip": "textattack.attack_recipes.HotFlipEbrahimi2017",
    "input-reduction": "textattack.attack_recipes.InputReductionFeng2018",
    "kuleshov": "textattack.attack_recipes.Kuleshov2017",
    "morpheus": "textattack.attack_recipes.MorpheusTan2020",
    "seq2sick": "textattack.attack_recipes.Seq2SickCheng2018BlackBox",
    "textbugger": "textattack.attack_recipes.TextBuggerLi2018",
    "textfooler": "textattack.attack_recipes.TextFoolerJin2019",
    "pwws": "textattack.attack_recipes.PWWSRen2019",
    "iga": "textattack.attack_recipes.IGAWang2019",
    "pruthi": "textattack.attack_recipes.Pruthi2019",
    "pso": "textattack.attack_recipes.PSOZang2020",
    "checklist": "textattack.attack_recipes.CheckList2020",
    "clare": "textattack.attack_recipes.CLARE2020",
    "a2t": "textattack.attack_recipes.A2TYoo2021",
    "textcheater": "textattack.attack_recipes.TextCheaterGuo2021",
    "hla": "textattack.attack_recipes.HardLabelMaheshwary2021",
    "hl_de": "textattack.attack_recipes.HardLabelDE2021",
    'mha': "textattack.attack_recipes.MHAGuo2023",
}
TEXTATTACK_DATASET_BY_MODEL = {
    #
    # LSTMs
    #
    "lstm-ag": (
        "data_temp/models/lstm-ag",
        ("ag_news", None, "test"),
    ),
    "lstm-imdb": ("data_temp/models/lstm-imdb", ("imdb", None, "test")),
    "lstm-mr": (
        "data_temp/models/lstm-mr",
        ("rotten_tomatoes", None, "test"),
    ),
    "lstm-sst2": ("data_temp/models/lstm-sst2", ("glue", "sst2", "validation")),
    "lstm-yelp": (
        "data_temp/models/lstm-yelp",
        ("yelp_polarity", None, "test"),
    ),
    #
    # CNNs
    #
    "cnn-ag": (
        "data_temp/models/cnn-ag",
        ("ag_news", None, "test"),
    ),
    "cnn-imdb": ("data_temp/models/cnn-imdb", ("imdb", None, "test")),
    "cnn-mr": (
        "data_temp/models/cnn-mr",
        ("rotten_tomatoes", None, "test"),
    ),
    "cnn-sst2": ("data_temp/models/cnn-sst", ("glue", "sst2", "validation")),
    "cnn-yelp": (
        "data_temp/models/cnn-yelp",
        ("yelp_polarity", None, "test"),
    ),
}


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC"""

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_corpus(path, clean=True, MR=True, encoding='utf8', shuffle=False, lower=False):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            if MR:
                label, sep, text = line.partition(' ')
                label = int(label)
            else:
                label, sep, text = line.partition(',')
                label = int(label) - 1
            if clean:
                text = clean_str(text.strip()) if clean else text.strip()
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text)

    return data, labels


def read_data(filepath, data_size=1000, lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if 'snli' in filepath:
        labeldict = {"contradiction": 0,
                     "entailment": 1,
                     "neutral": 2}
    elif 'mnli' in filepath:
        labeldict = {'0': 1, '1': 2, '2': 0}
    with open(filepath, 'r', encoding='utf8') as input_data:
        data = []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            data.append((collections.OrderedDict({'premise': premise, 'hypothesis': hypothesis}), labeldict[line[0]]))
        return data


def load_data_from_csv(dataset_path):
    texts, labels = read_corpus(dataset_path, clean=False, lower=False)
    texts, labels = texts[:], labels[:]
    data = list(zip(texts, labels))
    print(str(len(data)))
    return data


def textattack_create_model_form_path(model_name):
    if 'bert' in model_name:
        print(model_name)
        # Support loading models automatically from the HuggingFace model hub.
        import transformers
        model = transformers.AutoModelForSequenceClassification.from_pretrained(f'./data_temp/models/{model_name}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(f'./data_temp/models/{model_name}')
        model = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )
    elif model_name in TEXTATTACK_DATASET_BY_MODEL:
        # Support loading TextAttack pre-trained models via just a keyword.
        num_labels = 2
        if 'ag' in model_name:
            num_labels = 4
        model_path, _ = TEXTATTACK_DATASET_BY_MODEL[model_name]
        model = textattack.shared.utils.load_textattack_model_from_path(
            model_name, model_path
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(
            model, model.tokenizer
        )
    else:
        raise ValueError(f"Error: unsupported TextAttack model {model_name}")

    return model


def start_attack(attack_method, model_name, dataset, attack_number, file_logger, init_adv_examples, output_path,
                 goalfunction, start_index=0):
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # ___________________________________dataset_load_________________________________________
    dataset = load_data_from_csv('./data_temp/' + dataset)
    dataset = Dataset(dataset)

    # ___________________________________model_load_________________________________________
    if goalfunction[0] == 'untargeted':
        model_info = textattack_create_model_form_path(model_name)
        model_wrapper = UntargetedClassification(model_info, query_budget=goalfunction[1])
    elif goalfunction[0] == 'llm':
        from textattack.llm.llm_goalfunction import LLMGoalFunction
        model_wrapper = LLMGoalFunction(query_budget=goalfunction[1])
        # dataset = []
        # for example in init_adv_examples[1:]:
        #     dataset.append((example['original_text'], example['ground_truth_output']))
        # dataset = Dataset(dataset)
    else:
        raise Exception('wrong goal_function!')

    # ___________________________________Attack_load_________________________________________
    if attack_method == 'mha':
        from textattack.attack_recipes import MHAGuo2023
        attack = MHAGuo2023.build(model=model_wrapper, file_logger=file_logger, init_adv_examples=init_adv_examples)
    elif attack_method == 'qesa':
        from textattack.attack_recipes import QESAGuo2024
        attack = QESAGuo2024.build(model=model_wrapper, file_logger=file_logger)
    elif attack_method in ATTACK_RECIPE_NAMES:
        attack = eval(
            f"{ATTACK_RECIPE_NAMES[attack_method]}.build(model_wrapper)"
        )
    else:
        raise ValueError(f"Error: unsupported recipe {attack_method}")

    attack_args = AttackArgs(
        num_examples=attack_number,
        num_examples_offset=start_index,
        log_to_csv=output_path,
        disable_stdout=True
    )
    attack = textattack.Attacker(attack, dataset, attack_args)

    failed_attacks = 0
    skipped_attacks = 0
    successful_attacks = 0

    for result in attack.attack_dataset():
        if isinstance(result, FailedAttackResult):
            failed_attacks += 1
            continue
        elif isinstance(result, SkippedAttackResult):
            skipped_attacks += 1
            continue
        else:
            successful_attacks += 1

    rst_dict = {'succ_num': successful_attacks, 'failed_num': failed_attacks, 'skipped_num': skipped_attacks}

    return rst_dict


def load_initial_adversarial_example(path, start_index=0):
    import csv
    init_adv_rst = []
    example_count = 0
    with open(path, encoding='UTF-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['num_queries'].strip() == 'num_queries' or row['num_queries'].strip() == '' or row['result_type'] == "Skipped":
                continue
            else:
                if row['num_queries'].strip() == 'num_queries' or row['num_queries'].strip() == '':
                    continue
            init_adv_rst.append({'original_text': row['original_text'], 'perturbed_text': row['perturbed_text'],
                                 'result_type': row['result_type'], 'num_queries': float(row['num_queries'].strip()),
                                 'ground_truth_output': int(float(row['ground_truth_output']))})
    return init_adv_rst

# LLM
# DEFAULT_QUORA_ARGS = {'quora': {'data_size': 500-397, 'budget': 100, 'start_index': 397}}
# PROMPT_LEVEL = None
# LLM_LIST = ['vicuna-7b-v1.5', 'llama2-7b-chat', 'chatglm3-6b', 'dolly-v2-7b', 'mistral-7b-instruct-v0.2', 'gpt-3.5-turbo', 'claude-3-haiku-20240307']
#
#
# ALL_BUDGET = [DEFAULT_QUORA_ARGS]
# MODELS_LIST = ['claude-3-haiku-20240307']
# METHODS_LIST = ['qesa']

# models
DEFAULT_QUORA_ARGS = {'quora': {'data_size': 500, 'budget': 100, 'start_index': 0}}
PROMPT_LEVEL = None
LLM_LIST = ['vicuna-7b-v1.5', 'llama2-7b-chat', 'chatglm3-6b', 'dolly-v2-7b', 'mistral-7b-instruct-v0.2',
            'gpt-3.5-turbo', 'claude-3-haiku-20240307']


ALL_BUDGET = [
    {'mr': {'data_size': 500, 'budget': 150, 'start_index': 0}},
    # {'yelp': {'data_size': 500, 'budget': 200, 'start_index': 0}},
]

MODELS_LIST = ['bert']
METHODS_LIST = ['mha']

# TASK:
# RUN:
if __name__ == '__main__':
    # start_index = 0
    use_init_adv_examples = 1
    for DEFAULT_ARGS_V1 in ALL_BUDGET:
        datasets_name = list(DEFAULT_ARGS_V1.keys())
        for method in METHODS_LIST:
            for i in range(len(MODELS_LIST)):
                for j in range(len(datasets_name)):
                    model_name = f'{MODELS_LIST[i]}-{datasets_name[j]}'
                    print(f'Attack:{model_name}')
                    dataset_ = datasets_name[j]
                    budget = DEFAULT_ARGS_V1[datasets_name[j]]['budget']
                    attack_number = DEFAULT_ARGS_V1[datasets_name[j]]['data_size']
                    start_index = DEFAULT_ARGS_V1[datasets_name[j]]['start_index']
                    goal_function_type = 'untargeted'
                    if method == 'mha' and use_init_adv_examples:
                        init_adv_path = os.path.join('./data_temp/init_adv_examples',
                                                     f'{model_name}_leapattack-{attack_number + start_index}-{budget}.csv')
                        init_adv_examples = load_initial_adversarial_example(init_adv_path, start_index=start_index)
                    else:
                        init_adv_examples = None

                    if MODELS_LIST[i] in LLM_LIST:
                        goal_function_type = 'llm'
                    if PROMPT_LEVEL:
                        file_name = f'{model_name}_{method}-{attack_number + start_index}-{budget}_prompt{PROMPT_LEVEL}'
                    else:
                        file_name = f'{model_name}_{method}-{attack_number + start_index}-{budget}'
                    file_logger = FileLogger().build_new_log(
                        file_path=os.path.join('./data_temp/log', file_name + '.log'))
                    file_logger.insert_log('Start Logging...')
                    rst = start_attack(attack_method=method, model_name=model_name,
                                       dataset=dataset_,
                                       attack_number=attack_number,
                                       file_logger=file_logger,
                                       init_adv_examples=init_adv_examples,
                                       start_index=start_index,
                                       output_path=os.path.join('./data_temp/attack_results/',
                                                                file_name + '.csv'),
                                       goalfunction=(goal_function_type, budget))
                    torch.cuda.empty_cache()
