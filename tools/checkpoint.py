""" ADEF checkpoint implementation script """

from __future__ import annotations

import os
import gc
import pickle
import sys
import threading
from datetime import datetime
import functools
import hashlib
import tracemalloc

PATHS = ["./detectors", "./tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
from tools.log import Log
from tools.step_state import StepState

# TODO: Idea1 Versioning and timing can be added as
# TODO: framework measurements


# TODO: This is version 1 of checkpoint - to test
# TODO: basic functionalities. Once basis is working
# TODO: this will be converted to class based decorator
def checkpoint(step_name: str) -> object:
    """
    Decorator used in the experimentation
    Args:
        step_name   - step name used in the step state file naming (restoration and saving)
    Returns:
        <object>    - result of wrapper function execution result
    """

    def outer_wrapper(func) -> object:
        """
        Decorator used in the experimentation
        Args:
            func       - function object on top of which decorator is being used
        Returns:
            <object>    - result of wrapper function execution result
        """
        function_list = []

        def wrapper(cls, *args, **kwargs):
            # Perform check run status
            need_to_run = True
            # Resolve all required parameters correctly
            exp_path = (
                f"{cls['global']['path']}"
                if isinstance(cls, dict)
                else f"{cls.cfg['global']['path']}experiments/{cls.cfg['experiment']['path']}"
            )

            checkpoint_path = f"{exp_path}checkpoint/"
            main_exp_name = checkpoint_path.replace("/checkpoint/", "")
            main_exp_name = main_exp_name[main_exp_name.rfind("/") + 1 :]
            experiment_id = f"{main_exp_name}_summary"

            if isinstance(cls, dict):
                if "id" in cls["experiment"].keys():
                    experiment_id = cls["experiment"]["id"]
            else:
                experiment_id = cls.id

            a_path = (
                f"{exp_path}/" if isinstance(cls, dict) else cls.a_path
            )
            log = None
            if isinstance(cls, dict):
                for arg in args:
                    if isinstance(arg, Log):
                        log = arg
            else:
                log = cls.log

            file_path = f"{checkpoint_path}"
            file_path = file_path.replace("//", "/")
            file_name = f"{file_path}{experiment_id.lower()}_cp.pkl"

            # Enable visibility of the external static function_list variable
            nonlocal function_list

            # Read from file about this step
            if os.path.isfile(file_name):
                with open(file_name, "rb") as f2r:
                    function_list = pickle.load(f2r)

            # Hash function name and output path as a unique signature to ensure
            # that each step of different experiment will run only once.
            hash_name = hashlib.sha512(
                f"{func.__name__}{a_path}".encode("utf-8")
            ).hexdigest()

            if isinstance(cls, dict) and "ntr" in cls.keys() and cls["ntr"]:
                need_to_run = True
            else:
                # Check if function name has been already
                # written to a file
                if hash_name in function_list:
                    need_to_run = False

            if need_to_run:
                try:
                    # State restoration
                    if (
                        not isinstance(cls, dict)
                        and step_name
                        and step_name[-1].isdigit()
                    ):
                        step_number = int(step_name[-1])
                        if step_number > 1:
                            step_number -= 1
                            cls.best_models = StepState.restore(
                                cls.best_models,
                                cls.init_state_size,
                                f"{exp_path}checkpoint/{step_name[:-1]}{step_number}.pkl",
                            )
                    # Execute experiment step
                    func(cls, *args, **kwargs)

                    # Clear up memory after each experiment step
                    # to enable successful long run
                    gc.collect()

                    # Add function name
                    function_list.append(hash_name)

                    # Save current experiment step as done
                    with open(file_name, "wb") as f2w:
                        pickle.dump(function_list, f2w)

                    # Save object state
                    if (
                        not isinstance(cls, dict)
                        and step_name
                        and step_name[-1].isdigit()
                    ):
                        StepState.save(
                            cls.best_models,
                            f"{exp_path}checkpoint/{step_name}.pkl",
                        )
                except Exception as ex:
                    log.exception(ex)
                    raise ex
            else:
                # State restoration
                if not isinstance(cls, dict) and step_name and step_name[-1].isdigit():
                    cls.best_models = StepState.restore(
                        cls.best_models,
                        cls.init_state_size,
                        f"{exp_path}checkpoint/{step_name}.pkl",
                    )

                log.debug(f"Function: {func.__name__.upper()} already DONE")

        return wrapper

    return outer_wrapper


def trace_malloc(file_name: str) -> object:
    def inner(func: object) -> object:
        """
        Method performs memory tracing for the given function
        Args:
            func        - function to be wrapped
        Returns:
            <object>    - callable function that was wrapped
        """

        def wrapper(cls, *args, **kwargs):
            alloc_before = len(gc.get_objects())
            cls.log.debug(f"OTBefore :{alloc_before}")
            tracemalloc.start()

            func(cls, *args, **kwargs)

            snapshot = tracemalloc.take_snapshot()  # .dump(file_name)
            stats = snapshot.statistics("lineno", cumulative=True)

            tracemalloc.clear_traces()
            tracemalloc.stop()
            alloc_after = len(gc.get_objects())
            cls.log.debug(f"OTAfter :{alloc_after}")

            with open(file_name, "w+") as f2w:
                f2w.write("Allocation status:\n")
                f2w.write(f"\tBefore: {alloc_before}B\n")
                f2w.write(f"\tAfter : {alloc_after}B\n")
                f2w.write(f"\t---------------------\n")
                f2w.write(f"\tTotal : {alloc_after-alloc_before}B\n")

                for stat_line in stats:
                    f2w.write(f"{stat_line}\n")

        return wrapper

    return inner


def timing(func) -> object:
    """
    Method performs time measurements used in experimentation
    for each step of the computation
    Args:
        func        - function to be wrapped
    Returns:
        <object>    - callable function that was wrapped
    """

    def wrapper(cls, *args, **kwargs):
        # Initiate tme
        start = datetime.now()

        func(cls, *args, **kwargs)

        # Compute time
        stop = datetime.now()
        exec_time = (stop - start).total_seconds()

        # Resolve all required parameters correctly
        file_path = (
            f"{cls['global']['path']}"
            if isinstance(cls, dict)
            else f"{cls.cfg['global']['path']}experiments/{cls.cfg['experiment']['path']}"
        )

        main_exp_name = file_path[file_path.rfind("/") + 1 :]
        if not main_exp_name:
            main_exp_name = "data_sticher"
        experiment_id = f"{main_exp_name}_summary"

        file_path = f"{file_path}/timing/"

        if isinstance(cls, dict):
            if "id" in cls["experiment"].keys():
                experiment_id = cls["experiment"]["id"]
        else:
            experiment_id = cls.id

        log = None
        if isinstance(cls, dict):
            for arg in args:
                if isinstance(arg, Log):
                    log = arg
        else:
            log = cls.log

        if not isinstance(cls, dict) and hasattr(cls, "id"):
            class_name = experiment_id
        else:
            class_name = "data_sticher"
            # cls.__class__.__name__.lower()

        file_name = f"{file_path}{class_name}_tm.csv"

        # Save measurements to a file
        with open(file_name, "w") as f2w:
            f2w.writelines([f"{class_name},{func.__name__},{str(exec_time)}"])

        log.debug(f"Exec time: {exec_time:.2f} seconds")

    return wrapper


class Timer:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, cls, *args, **kwargs):
        start = datetime.now()
        # print(f"Call {self.num_calls} of {self.func.__name__!r}")
        cls.func(cls, *args, **kwargs)
        exec_time = (start - datetime.now()).total_seconds()
        # Add to object timing information
        # args[0].timing[self.func.__name__] = exec_time

        # Save measurements to a file
        with open(args[0].id + "_tm.csv", "w") as f2w:
            f2w.writelines(
                [args[0].id + "," + self.func.__name__ + "," + str(exec_time)]
            )


# TODO: Fix this class
class Checkpoint:
    """
    Class defining given application checkpoint behavior
    """

    def __init__(self, cfg: dict, logger: Log) -> Checkpoint:
        """
        Args:
            cfg     - configuration object
            logger  - logger object
        Returns:
            <Checkpoint>    - instance of an Checkpoint class
        """
        self.log = logger
        self.cfg = cfg
        self.mutex = threading.Lock()
        # Cache that will store application checkpoints
        self.cache = {}

    def load(self, file_name) -> bool:
        """
        Method loads given checkpoints from the disk
        through pickle deserialization
        Args:
        Returns:
            <bool>
        """
        with open(self.cfg["file"], "rb") as f2r:
            self.cache = pickle.load(f2r)

    def save(self) -> None:
        """
        Method saves current checkpoint
        """
        with open(self.cfg["file"], "wb") as f2w:
            pickle.dump(self.cache, f2w)

    def func_name():
        import traceback

        return traceback.extract_stack(None, 2)[0][2]

    def func_name1():
        import inspect

        print(inspect.stack()[0][3])

    # TODO: This can be implemented as decorator for other Python functions
    # TODO: as post function activity
    def add(self, function_name: str) -> None:
        """
        Method adds completion of given checkpoint
        Args:
            function_name   - name of the function to be checked
        Returns:
            None
        """

        # 1. Extract from the stack one level above the caller function name to be written to local cache
        # 2. Save it in cache
        pass

    # TODO: This can be implemented as decorator for other Python functions
    def check(self, function_name: str) -> bool:
        """
        Method check if given function has been executed
        on previous run.
        Args:
            function_name   - name of the function to be checked
        Returns:
            <bool>  - True if function has been found in cache
                      False otherwise
        """
        result = False
        # Function names extraction
        with self.mutex:
            if function_name in self.cache:
                result = True

        return result
