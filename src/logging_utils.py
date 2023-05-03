import contextlib
import logging


class OutputLogger:
    def __init__(
        self,
        log_path,
        level_default="INFO",
        stream_handler_format="%(message)s",
        file_handler_format="%(asctime)s %(levelname)s:%(message)s",
    ):
        logging.basicConfig(
            level=logging.INFO,
            format=stream_handler_format,
            handlers=[
                logging.FileHandler(log_path, mode="w"),
                logging.StreamHandler(),
            ],
        )

        log = logging.getLogger()

        # Remove existing file handler
        for hdlr in log.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                log.removeHandler(hdlr)

        # Replace with new handler
        filehandler = logging.FileHandler(log_path, "w")
        formatter = logging.Formatter(file_handler_format)
        filehandler.setFormatter(formatter)
        log.addHandler(filehandler)

        self.logger = log
        self.name = self.logger.name
        self.level = getattr(logging, level_default)
        self._redirector = contextlib.redirect_stdout(self)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def exception(self, msg):
        self.logger.exception(msg)

    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)
