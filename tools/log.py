"""Dual stream logging module for outputting log to screen and file at the same time"""


# Logging facility
import logging
import sys
import time
from enum import StrEnum

# Threading
import threading

# Exception tracing
import traceback
from logging.handlers import RotatingFileHandler

# from colorama import Fore, Back, Style
from colorama import init


class BColors(StrEnum):
    """
    Background colors definition
    """

    # Coloring of the message level
    # Info - green on yellow
    # Debug - grey on black
    # Waring - yellow on black
    # Error - red on yellow
    # Exception - white on blue
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    CRIT = "\033[101m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


class Log:
    """
    Logger - wrapper class for Python log facility
    that sends messages to file and standard output.
    This class is thread safe.
    """

    def __init__(self, log_file_name: str, log_level=logging.DEBUG, log_size=0) -> None:
        """
        CTOR - configures logger
        Args:
            log_file      - log file where all logging information will be dumped
            log_level     - logging level. Default value is set to logging.DEBUG
            log_size      - circular buffer size - default value is 5MB of max log
        Returns:
            <None>
        """
        # Create mutex for multi-threaded log processing
        self.mutex = threading.Lock()
        # Initialize colorama for colored output on Windows
        init()
        self.curr_logging_level = log_level

        with self.mutex:
            start_idx = 0
            stop_idx = len(log_file_name) - 1
            if "/" in log_file_name:
                start_idx = log_file_name.rindex("/") + 1
            if "." in log_file_name:
                stop_idx = log_file_name.rindex(".")

            log_name = log_file_name[start_idx:stop_idx]

            log_file = log_file_name
            if "." not in log_file:
                log_file += ".log"

            # Log size in megabytes it is divided by 2 to keep all log size in
            # accordance to user specification
            self.log_size = int(log_size / 2)
            # Create logger with given log name, log level and  assign
            self.logger = logging.getLogger(log_name)
            self.logger.setLevel(log_level)

            # Create file handler which logs even debug messages
            # In case of limit use rotating file handler
            if self.log_size == 0:
                self.file_handler = logging.FileHandler(log_file)
            else:
                self.file_handler = RotatingFileHandler(
                    log_file,
                    mode="a",
                    maxBytes=self.log_size * 1024 * 1024,
                    backupCount=1,
                    encoding=None,
                    delay=0,
                )

            self.file_handler.setLevel(log_level)

            # File formatter section
            log_format = "%(asctime)-15s.%(msecs).03d %(levelname)-8s %(message)s"
            log_date_format = "%Y/%m/%d %H:%M:%S"

            self.f_formatter = logging.Formatter(
                fmt=log_format, datefmt=log_date_format
            )
            self.file_handler.setFormatter(self.f_formatter)

            # Output console handler with a higher log level
            log_output_format = (
                f"%(asctime)-15s.%(msecs).03d{BColors.ENDC} "
                f"{BColors.OKBLUE}%(levelname)-6s{BColors.ENDC} %(message)s"
            )
            log_output_date_format = f"{BColors.BOLD}%Y/%m/%d %H:%M:%S"
            self.o_formatter = logging.Formatter(
                fmt=log_output_format, datefmt=log_output_date_format
            )
            self.channel_handler = logging.StreamHandler()
            self.channel_handler.setLevel(log_level)
            self.channel_handler.setFormatter(self.o_formatter)

            # Add the handlers to the logger
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.channel_handler)

    def set_format(self, log_level: object) -> None:
        """
        Method changes output formatting based on the current logging level
        Args:
            log_level: logging level name
        Returns:
            <None>
        """
        if self.curr_logging_level == log_level:
            return

        log_color = BColors.OKCYAN
        if logging.INFO == log_level:
            log_color = BColors.OKCYAN
        elif log_level in [logging.WARNING, logging.WARN]:
            log_color = BColors.WARNING
        elif logging.DEBUG == log_level:
            log_color = BColors.OKBLUE
        elif logging.ERROR == log_level:
            log_color = BColors.FAIL
        elif logging.CRITICAL == log_level:
            log_color = BColors.CRIT

        log_output_format = (
            f"%(asctime)-15s.%(msecs).03d{BColors.ENDC} "
            f"{log_color}%(levelname)-8s{BColors.ENDC} %(message)s"
        )

        log_output_date_format = f"{BColors.BOLD}%Y/%m/%d %H:%M:%S"
        self.o_formatter = logging.Formatter(
            fmt=log_output_format, datefmt=log_output_date_format
        )
        self.channel_handler.setFormatter(self.o_formatter)
        self.curr_logging_level = log_level

    def close(self) -> None:
        """
        Close logger and flush out all messages
        Input parameter:
            None
        Output parameter:
            None
        """
        if self.file_handler is not None:
            self.file_handler.flush()
        if self.channel_handler is not None:
            self.channel_handler.flush()
            self.channel_handler.close()
        if self.file_handler is not None:
            self.file_handler.close()

    def __set_terminator(self, kwargs: dict) -> None:
        """
        Method sets correctly terminator
        Args:
            kwargs: key-world argument dictionary containing logger configuration
        Returns:
            <None>
        """
        if "end" in kwargs.keys():
            self.channel_handler.terminator = kwargs["end"]
        else:
            self.channel_handler.terminator = "\n"

    def __stringify(self, *args) -> str:
        """
        Method converts list of arguments into single string
        Args:
            args - list of arguments to be stringified
        Returns:
            <string>    - string formatted message ready for log
        """
        msg = ""
        if "\r" in args[0]:
            msg = "\r"

        for argument in args[0]:
            if not str(argument):
                continue
            if not isinstance(argument, str):
                msg += str(argument) + " "
            else:
                msg += argument
                if not argument.endswith(" "):
                    msg += " "
        return msg

    def info(self, *args, **kwargs) -> None:
        """
        Wrapper for info method of Python logger
        Args:
            args    - list of arguments to be logged
            kwargs  - dictionary containing key/value pair arguments
        Returns:
            None
        """
        with self.mutex:
            msg = self.__stringify(args)
            if self.logger is not None:
                self.set_format(logging.INFO)
                self.__set_terminator(kwargs)
                self.logger.info(msg)
            else:
                print(msg)

    def warn(self, *args, **kwargs) -> None:
        """
        Wrapper for warn method of Python logger
        Args:
            args    - list of arguments to be logged
            kwargs  - dictionary containing key/value pair arguments
        Returns:
            None
        """
        with self.mutex:
            msg = self.__stringify(args)
            if self.logger is not None:
                self.set_format(logging.WARNING)
                self.__set_terminator(kwargs)
                self.logger.warning(msg)
            else:
                print(msg)

    def debug(self, *args, **kwargs) -> None:
        """
        Wrapper for debug method of Python logger
        Args:
            args    - list of arguments to be logged
            kwargs  - dictionary containing key/value pair arguments
        Returns:
            None
        """
        with self.mutex:
            msg = self.__stringify(args)
            if self.logger is not None:
                self.set_format(logging.DEBUG)
                self.__set_terminator(kwargs)
                self.logger.debug(msg)
            else:
                print(msg)

    def error(self, *args, **kwargs) -> None:
        """
        Wrapper for error method of Python logger
        Args:
            args    - list of arguments to be logged
            kwargs  - dictionary containing key/value pair arguments
        Returns:
            None
        """
        with self.mutex:
            msg = self.__stringify(args)
            if self.logger is not None:
                self.set_format(logging.ERROR)
                self.__set_terminator(kwargs)
                self.logger.error(msg)
            else:
                print(msg)

    def critical(self, *args, **kwargs) -> None:
        """
        Wrapper for critical method of Python logger
        Args:
            args    - list of arguments to be logged
            kwargs  - dictionary containing key/value pair arguments
        Returns:
            None
        """
        with self.mutex:
            msg = self.__stringify(args)
            if self.logger is not None:
                self.set_format(logging.CRITICAL)
                self.__set_terminator(kwargs)
                self.logger.critical(msg)
            else:
                print(msg)

    def exception(self, external_exception) -> None:
        """
        Logs Python exception with full detail
        Args:
            external_exception - exception to be logged with its full trace stack
        Returns:
            None
        """
        with self.mutex:
            if self.logger is not None:
                limit = None
                self.set_format(logging.CRITICAL)
                etype, value, trace_back = sys.exc_info()
                exception_msg = traceback.format_exception(
                    etype, value, trace_back, limit
                )
                self.logger.critical("".join(exception_msg))
            else:
                print(external_exception)


if "__main__" == __name__:
    log2F = Log("test.log")
    # Colors test
    X = 5
    log2F.info(f"Test Info {X}")
    log2F.warn(f"Test Warn {X}")
    log2F.debug(f"Test debug {X}")
    log2F.error(f"Test error {X}")
    log2F.critical(f"Test critical {X}")
    # Progress

    for i in range(1, 20):
        log2F.debug("#" * i, end="\r")
        time.sleep(0.1)

    for i in range(1, 100):
        log2F.debug(f"test {i}%", end="\r")
        time.sleep(0.1)

    log2F.close()
