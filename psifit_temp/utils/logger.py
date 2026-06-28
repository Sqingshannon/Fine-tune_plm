import sys
import logging

class Logger(object):
    def __init__(self, logfile=None, level=logging.DEBUG):
        '''
        logfile: pathlib object
        '''
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s\t%(message)s", "%Y-%m-%d %H:%M:%S")

        for hd in self.logger.handlers[:]:
            self.logger.removeHandler(hd)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        if logfile is not None:
            logfile.parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(logfile, 'w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)


    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)