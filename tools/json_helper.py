import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    Class that helps convert correctly to output string numpy data inside
    JSON object.
    """

    def default(self, obj) -> str:
        """
        Method helps convert larger numpy types into Python default types
        during JSON object conversion to a string.

        Threat this method with caution and understanding that the displayed
        values will be cast into shorter type - this might lead into
        incorrect outcome of the print. In doubt please refer to
        raw data debugging to help triage this problem or remap types
        before dumping json to string format to avoid conversion exceptions.
        """
        if isinstance(obj, np.integer) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.floating) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, str) and "\\/" in obj:
            return obj.replace("\\/", "")
        return json.JSONEncoder.default(self, obj)
