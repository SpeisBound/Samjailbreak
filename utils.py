import torch
import logging
from logging.handlers import HTTPHandler
import os
from secrets import token_urlsafe
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# aitest data
DATASET_SAVE_PATH = 'data/aitest/datasets'
zip_cache_path = 'data/zip_cache'
# bll url
BLL_URL = {'host': 'http://127.0.0.1:5001', 'logger_host': '127.0.0.1:5001', 'task_log_url': '/task/history/task_log',
           'model_upload_verify_url': '/upload/model/verify', 'dataset_upload_verify_url': '/upload/dataset/verify'}

# text_attack configuration
USE_ENCODING_PATH = "data/aitest/models/tfhub_modules/063d866c06683311b44b4992fd46003be952409c"
TEXTATTACK_CACHE_DIR = 'data/aitest/models/textattack'


DATA_PATH = '/data'


################################################################################################
class FileLogger:
    def __init__(self):

        self.logger = logging.getLogger('aitest-filelogger')
        self.logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)
        self.file_handler = None

    def build_new_log(self, file_path):
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)
        self.file_handler = file_handler
        self.logger.addHandler(self.file_handler)
        return self

    def insert_log(self, content=None):
        if content:
            self.logger.info(content)

    def finish_log(self):
        self.logger.removeHandler(self.file_handler)


class HttpLogger:

    def __init__(self, log_path, task_udc=None):
        self.log_path = log_path
        self.task_udc = task_udc

        self.logger = logging.getLogger('aitest-textattack')
        self.logger.setLevel(logging.INFO)

        url = BLL_URL['task_log_url']
        log_handler = HTTPHandler(host=BLL_URL['logger_host'], url=f'{url}?task_udc={self.task_udc}', method='GET')
        log_handler.setLevel(level=logging.INFO)
        self.fh = log_handler

    def build_new_log(self):
        self.logger.addHandler(self.fh)
        return self

    def insert_log(self, content=None):
        if content:
            self.logger.info(content)

    def finish_log(self):
        self.logger.removeHandler(self.fh)

    @staticmethod
    def log_attack_result(number, result):
        content = "-" * 45 + " Result " + str(number) + " " + "-" * 45 + "\n"+result.__str__(color_method="file") + '\n'
        return content


def get_size(path, show_bytes=True):
    size = 0
    if os.path.isfile(path):
        size = os.path.getsize(path)
    elif os.path.isdir(path):

        for root, dirs, files in os.walk(path):
            for file in files:
                size += os.path.getsize(os.path.join(root, file))
    if show_bytes:
        return int(size)

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.1f %s" % (size, x)
        size /= 1024.0
    return size


def path_generate(ownership='aitest', table_name='files', file_name='', udc_suffix=True, extension_name='.csv'):
    if udc_suffix:
        return os.path.join(DATA_PATH, ownership, table_name, file_name+'_' + token_urlsafe(8)+extension_name).replace('\\', '/')
    else:
        return os.path.join(DATA_PATH, ownership, table_name, file_name+extension_name).replace('\\', '/')


def dir_generate(ownership='aitest', table_name='files', dir_name='', udc_suffix=True):
    if udc_suffix:
        return os.path.join(DATA_PATH, ownership, table_name, dir_name+'_' + token_urlsafe(8)).replace('\\', '/')
    else:
        return os.path.join(DATA_PATH, ownership, table_name, dir_name).replace('\\', '/')


def to_zip(archive_name, folder_to_archive, fm='zip'):
    if not os.path.exists(zip_cache_path):
        os.mkdir(zip_cache_path)
    archive_name = f'{zip_cache_path}/{archive_name}'
    shutil.make_archive(archive_name, fm, folder_to_archive)
    return archive_name


