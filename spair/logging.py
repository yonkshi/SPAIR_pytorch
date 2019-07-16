import logging
import os
import sys
import requests


def log(*args):
    logger = logging.getLogger('spair')
    args = [str(arg) for arg in args]
    logger.info(*args)

def record_scalar(t, name, group):
    pass

def record_image():
    pass

def telegram_yonk(message):
    # Telegram notify myself
    param = dict(
        chat_id=390311059,
        disable_web_page_preview=1,
        text= message,
    )
    url = 'https://api.telegram.org/bot818353417:AAGN8Jt25kIUy8IaQxGt9MKITpzDqDkao3k/sendMessage'
    requests.get(url, param)

def init_logger(log_path):
    logger = logging.getLogger('spair')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    run_log = os.path.join(log_path,'run.log')
    fh = logging.FileHandler(run_log, mode='w')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(MyFormatter())
    fh.setFormatter(MyFormatter())

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

class MyFormatter(logging.Formatter):
    def format(self, record):
        # optionally append
        if len(record.args) > 0:
            args = ' '.join(record.args)
            record.msg += ': \t' + args
            record.args = () # clear out args
        return super().format(record)