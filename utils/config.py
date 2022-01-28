from ast import literal_eval
from configparser import ConfigParser


def _literal_eval_new(str):
    result = str
    try:
        result = literal_eval(str)
    except (SyntaxError, ValueError):
        pass
    return result


class Config(object):
    '''
    类名：Config
    参数:
        path (str):
            配置文件路径
    '''

    def __init__(self, path):
        self._config = ConfigParser()
        self._config.read(path)

        self._kwargs = dict((option, _literal_eval_new(value))
                            for section in self._config.sections()
                            for option, value in self._config.items(section))
        self._option2section = dict((option, sect)
                                    for sect in self._config.sections()
                                    for option, _ in self._config.items(sect))

    def __repr__(self):
        s = "-" * 18 + "-+-" + "-" * 30 + "\n"
        s += f"{'Param':18} | {'Value':^30}\n"
        s += "-" * 18 + "-+-" + "-" * 30 + "\n"
        for _, (option, value) in enumerate(self._kwargs.items()):
            s += f"{option:18} | {str(value):^30}\n"
        s += "-" * 18 + "-+-" + "-" * 30 + "\n"

        return s

    def __getattr__(self, attr):
        return self._kwargs.get(attr, None)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        kwargs = {key: value for key,
                                 value in kwargs.items() if value is not None}
        self._kwargs.update(kwargs)
        for option, value in kwargs.items():
            if option in self._option2section:
                self._config.set(
                    self._option2section[option], option, str(value))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            self._config.write(f)
