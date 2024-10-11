from __future__ import annotations

import importlib
import inspect
import warnings
import re


class ADM_Importer:
    """
    Anomaly Detection Model Importer class implementation
    """
    @staticmethod
    def get_class(import_string: str) -> object:
        """
        Method imports
        """
        # Resolve import strings
        reg_exs = [r"import (?P<class_name>[\w]+)",
                   r"from (?P<module_path>[\w.]+) import (?P<class_name>[\w]+)"]
        my_class = None
        match = None

        for regex in reg_exs:
            match = re.match(regex, import_string, re.MULTILINE)

            if match is not None:
                break
        if match is not None:
            module = importlib.import_module(match.group("module_path"))
            my_class = getattr(module, match.group("class_name"))

        return my_class

    @staticmethod
    def resolve_function_names(model: object,
                               function_names: list) -> dict:
        """
        Method resolves function name
        Args:
            model: model object from where given function name will be resolved
            function_names: list of function names to be resolved from the model
        Returns:
            <dict> - extracted function for function name from the compatible model
        """
        # Get all attributes of the model

        # Ignore some warnings that might pop-up during class search
        # Those warnings are specific to searched classes not the inspect library
        # The warning filter is applied locally with use of context manager thus
        # it is not impacting the rest of the code or client code.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*class_weight_.*")

            attributes = [x for x in inspect.getmembers(model, predicate=inspect.ismethod)
                          if not x[0].startswith("_") and
                            x[0] in function_names]

        # Filter only the function objects
        # Use dictionary comprehension to get the signatures of the specified functions
        functions = {func_name: [attr[1], str(inspect.signature(attr[1]))] for func_name in function_names
                     for attr in attributes if func_name == attr[0]}

        return functions

    @staticmethod
    def get_model(model_import: str,
                  function_list: list = ["fit_predict", "fit", "predict"],
                  logger: object = None) -> object:
        """
        Method imports anomaly detection model
        Args:
            model_import: model import Python string
            function_list: name of the functions to resolve from the object
        Return:
            <object>: anomaly detection object
        """
        # Importing library by name from configuration file
        imported_class = ADM_Importer.get_class(model_import)
        model_class = imported_class()

        # TODO: Check function signatures in model class
        functions = ADM_Importer.resolve_function_names(model_class, function_list)

        return model_class, functions
