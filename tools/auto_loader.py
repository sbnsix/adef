from __future__ import annotations

import logging
import os
import importlib
import glob
import inspect


class AutoLoader:
    """
    Autoloader class that defines methods to automatically load the Python modules
    and classes to memory.
    """

    @staticmethod
    def load(file_path: str, file_mask: str, args: list) -> dict:
        """
        Generic method to load Python modules from given folder/file mask setting and
        initialize the module classes with generic list of arguments
        Args:
            file_path: file path for the Python files to be loaded
            file_mask: file mask that is used to search Python modules
            args: list of argument for the object to be initialized with
        Returns:
            <dict> - dictionary containing list of the
        """
        modules_cache = {}
        # Find experiment files inside exp folder
        file_modules = glob.glob(f"{file_path}/{file_mask}")
        x = os.getcwd()
        for file_module in file_modules:
            module_name = file_module[file_module.rindex("\\") + 1 : -3]
            # Automatically import module
            module = importlib.import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            for name, obj in classes:
                if obj.__module__ == module.__name__:
                    # Initialize only class and store class methods in dictionary
                    if name not in modules_cache.keys():
                        # Uniform class names to lower as it is difficult to guess the correct name
                        # after dynamic loading.
                        class_name = name.lower()

                        if "_" in class_name:
                            class_name = class_name[class_name.find('_')+1:]
                        try:
                            modules_cache[class_name] = obj(*args)
                        except Exception as ex:
                            logging.error(f"Unable to load Module:[{file_module}]:Class[{class_name}]")
                            logging.exception(ex)

        return modules_cache
