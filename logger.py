class Logger(object):

    def __init__(self,name = 'main', show_in_console = True):
        self.name = name
        self.show_in_console = show_in_console

    def create_logger(self):
        import logging
        import sys

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        fh = logging.FileHandler(self.name+ '.log')

        ch.setLevel(logging.DEBUG)
        formatter_log_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        formatter_console = logging.Formatter('*** %(levelname)s - %(message)s')

        ch.setFormatter(formatter_console)
        fh.setFormatter(formatter_log_file)

        root.addHandler(fh)

        if self.show_in_console:
            root.addHandler(ch)
        return root


'''
     let us test this logging funtion
        - log file is saved in output folder
        - option to print in terminal/console
'''

if __name__ == "__main__":
    logger = Logger('simulator', show_in_console = True)
    root = logger.create_logger()

    for i in range(10):
        root.info(str(i))
        if i == 5:
            root.debug(str('kkaka'))