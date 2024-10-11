"""Decorator module"""


from __future__ import annotations

from datetime import datetime
import inspect

# Import for system use
import os
import sys
import traceback
import warnings

# Tabular formatting display
from tabulate import tabulate

# Local modules
PATHS = ["./"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if not path in sys.path:
        sys.path.insert(1, path)


def AccessAttributeClass(cls):
    """
    Metaclass helper used in conjunction with decorator for function exception wrapping
    It allows function decorator access filed in the wrapped object.
    """

    class __metaclass__(type):
        """
        Meta class definition for class decoration
        """

        def __new__(cls, name, bases, dict_) -> object:
            """
            Args:
                name
                bases
                dict_
            Returns:
                <object>    - wrapped class
            """
            return type.__new__(cls, name, bases, dict_)

    return cls


def ExceptionTrace(func):
    """
    Decorator for exception catching in the object methods.
    Since local logging is used it needs to be added
    with AccessAttributeClass decorator on the class level.
    """

    def decorated(*args: list, **kwargs: dict) -> object:
        """
        Method wraps up input function for exception trace.
        Args:
            args    - list of function to be decorated arguments
            kwargs  - dictionary of key/value function parameter parirs
        Returns:
            <object>    - decorated function
        """
        # Make sure that function remains the same
        decorated.__name__ = func.__name__
        test_function = False
        if func.__name__.lower().startswith("test_"):
            test_function = True
        class_func_name = ""
        if args[0].__class__ is not None:
            class_func_name += args[0].__class__.__name__
        class_func_name += "." + func.__name__

        try:
            if test_function:
                info = f"TM: {class_func_name}"
                if hasattr(args[0], "log") and args[0].log is not None:
                    args[0].log.info(info)
                else:
                    print(info)
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(args[0], "stats") and args[0].stats is not None:
                stats = getattr(args[0].stats, "addException", None)
                if callable(stats):
                    args[0].stats.addException(e)
            if hasattr(args[0], "log") and args[0].log is not None:
                args[0].log.exception(e)
            else:
                limit = None
                etype, value, tracebuffer = sys.exc_info()
                ex_msg = "".join(
                    traceback.format_exception(etype, value, tracebuffer, limit)
                )
                print(ex_msg)

            # Raise exception here, so it can be correctly tracked
            raise e
        finally:
            if test_function:
                info = f"TM: {class_func_name} completed"
                if hasattr(args[0], "log") and args[0].log is not None:
                    args[0].log.info(info)
                else:
                    print(info)

    return decorated


def WarningTrace(warning_category):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger_loc = print
            if hasattr(args[0], "log") and args[0].log is not None:
                logger_loc = args[0].log.warn

            with warnings.catch_warnings(record=True) as caught_warnings:
                # Temporarily change warnings to raise an exception
                warnings.simplefilter("error", warning_category)
                try:
                    return func(*args, **kwargs)
                except warning_category as e:
                    logger_loc(f"Caught {warning_category.__name__} as exception: {e}")

            # If we got here, no warning was raised
            if not caught_warnings:
                logger_loc(f"No {warning_category.__name__} caught")

        return wrapper

    return decorator

def Arguments(func):
    """
    Decorator that captures input parameters of given function.
    It is used frequently with CTORs, and other initializaiton functions.
    Inside tabulate formattig is used to improve overall reporting
    capability.
    Since local logging is used it needs to be added
    with AccessAttributeClass decorator on the class level.
    """

    def decorated(*args: list, **kwargs: dict) -> object:
        """
        Method wraps up input function for argument debugging.
        Args:
            args    - list of function to be decorated arguments
            kwargs  - dictionary of key/value function parameter parirs
        Returns:
            <object>    - decorated function
        """
        # Make sure that function remains the same
        decorated.__name__ = func.__name__

        # Capture argument specification
        arg_spec = inspect.getfullargspec(func)
        obj_name = str(args[0]).split(" ")[0][1:]
        args_report = f"\n  Object   :[{obj_name}]\n  Function :[{func.__name__}]\n"

        # Preventing into going to trouble in CTOR where log is not initialized yet
        if func.__name__ == "__init__":
            print(f"Cannot scan and display arguments from CTOR in {args_report}")
            return func(*args, **kwargs)

        rep = []

        # Iterate through args
        for i in range(1, len(args)):
            rep.append([arg_spec[0][i], args[i]])

        # Iterate through kwargs
        for k, v in kwargs.items():
            rep.append([k, str(v)])

        if len(rep) > 0:
            args_report += "Input arguments:\n"

            # Build argument table in tabulate
            args_report += tabulate(
                rep, headers=["Parameter", "Value"], tablefmt="orgtbl"
            )
        else:
            args_report += "Input arguments: None\n"

        args[0].log.info(args_report)

        return func(*args, **kwargs)

    return decorated


def ServiceTrace(func):
    """
    Decorator to be used on service execution tracing
    """

    def decorated(*args: list, **kwargs: dict) -> object:
        """
        Method wraps up input function for service debugging.
        Args:
            args    - list of function to be decorated arguments
            kwargs  - dictionary of key/value function parameter parirs
        Returns:
            <object>    - decorated function
        """
        # Make sure that function remains the same
        decorated.__name__ = func.__name__
        # Add timestamp
        args[0].log.info(f"Executing: {func.__class__.__name__}->{func.__name__}")
        result = func(*args, **kwargs)
        args[0].log.info(
            f"Executing: {func.__class__.__name__}->{func.__name__} completed"
        )
        return result

    return decorated


def Timer(func):
    """
    Decorator used for function or method time performance measurement.
    """

    def decorated(*args: list, **kwargs: dict) -> object:
        """
        Method wraps up input function for time measurement.
        Args:
            args    - list of function to be decorated arguments
            kwargs  - dictionary of key/value function parameter pairs
        Returns:
            <object>    - decorated function
        """
        start = datetime.now()
        result = func(*args, **kwargs)
        stop = datetime.now()

        class_func_name = ""
        if args[0].__class__ is not None:
            class_func_name += args[0].__class__.__name__
        class_func_name += "." + func.__name__

        line = "---------------------------------------------------------"
        args[0].log.info(line)
        exec_time = (stop - start).total_seconds()
        args[0].log.info(f"{class_func_name} execution time: {exec_time:.2f} seconds")
        args[0].log.info(line)

        return result

    return decorated


def timing_debug(func) -> object:
    """
    Method performs time measurements used in experimentation
    for each step of the computation
    Args:
        func        - function to be wrapped
    Returns:
        <object>    - callable function that was wrapped
    """

    def wrapper(cls, *args, **kwargs):
        # Measurement start
        start = datetime.now()

        # Wrapped function execution
        func(cls)

        # Compute time
        stop = datetime.now()
        ts = (stop - start).total_seconds()
        cls.log.debug(f"{ts:.2f} seconds")

        # Save measurements to a file
        # with open(cls.id + "_tm.csv", "w") as f2w:
        #    f2w.writelines([cls.id + "," + func.__name__ + "," + str(ts)])

    return wrapper
