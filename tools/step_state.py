""" Simulation step data preservation module"""

from __future__ import annotations
import os
import sys
import pickle
import hmac
import hashlib


class StepState:
    """Class describing serialization method used between simulation steps
    More details on serialization can be found at:
    https://docs.python.org/3/library/pickle.html
    """

    # TODO: Fix this by generating key from underlying OS PKI/user certificate
    # The secret key for the HMAC signature
    secret_key = b"secret"

    @staticmethod
    def get_size(obj) -> int:
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(
                [StepState.get_size(k) + StepState.get_size(v) for k, v in obj.items()]
            )
        # elif isinstance(obj, list):
        #    size += sum(ObjTool.get_size(i) for i in obj)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([StepState.get_size(i) for i in obj])
        return size

    @staticmethod
    def serialize(state: object, file_name: str) -> None:
        """
        Method serialize the Python object - simulation state into a file
        Args:
            state: object to be serialized.
            file_name: name of the file name.
        Returns:
            <None>
        """
        # Pickle the object and get the HMAC signature
        pickled_data = pickle.dumps(state)
        signature = hmac.new(StepState.secret_key, pickled_data, hashlib.sha3_512).digest()

        # Write signature and
        with open(file_name, "wb") as f2w:
            f2w.write(signature)
            f2w.write(pickled_data)

    @staticmethod
    def deserialize(file_name: str) -> object:
        """
        Method serialize the Python object - simulation state into a file
        Args:
            file_name: name of the file name from where data will be retrieved.
        Returns:
            <object> - deserialized
        """
        data = None
        with open(file_name, "rb") as f2r:
            saved_signature = f2r.read(hashlib.sha3_512().digest_size)
            data_content = f2r.read()

            # Check the HMAC signature before unpickling the data
            calc_signature = hmac.new(
                StepState.secret_key, data_content, hashlib.sha3_512
            ).digest()
            if hmac.compare_digest(saved_signature, calc_signature):
                # If the signatures match, it's safe to unpickle the data
                data = pickle.loads(data_content)
            else:
                print("The data has been tampered with!")

        return data

    @staticmethod
    def save(data: object, file_name: str) -> None:
        """
        Method saves experiment best result state to disk
        Args:
            data: state to be saved
            file_name: name of the file where data will be written
        Returns:
            <None>
        """
        StepState.serialize(
            data,
            file_name,
        )

    @staticmethod
    def restore(initial_state: object, init_state_size: int, file_name: str) -> object:
        """
        Method restores experiment best result state
        in case of step failure or debugging (to save time)
        Args:
            initial_state:
            init_state_size:
            file_name: name of the step (step3, step4, step5)
        Returns:
            <object> - deserialized object
        """
        if StepState.get_size(initial_state) <= init_state_size and os.path.isfile(
            file_name
        ):
            state_data = StepState.deserialize(file_name)
            return state_data

        return initial_state
