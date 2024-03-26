from io import TextIOWrapper
import os
import time

class Utils(object):
    log_file: TextIOWrapper = None

    @staticmethod
    def log(*args, **kwargs):
        print(*args, **kwargs)
        if Utils.log_file is not None:
            print(*args, **kwargs, file=Utils.log_file)

    @staticmethod
    def set_log_file(file_path, append=False):
        if Utils.log_file is not None:
            Utils.log_file.close()

        if file_path is None:
            Utils.log_file = None
            return
        
        if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
        Utils.log_file = open(file_path, 'a' if append else 'w', encoding='utf-8')

    @staticmethod
    def get_time_str():
        return time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    
    @staticmethod
    def get_root_path():
        return os.path.dirname(os.path.abspath(__file__))