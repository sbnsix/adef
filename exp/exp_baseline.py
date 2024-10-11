""" Configuration builder for baseline experiment """

from exp.base_exp import BaseExperimentConfig


class ExpBaseline(BaseExperimentConfig):
    """
    Factory defining configuration modifications for the
    given experiment
    """

    def get(self, scafold: bool, *args) -> list:
        """
        Method generates baseline experiment configuration
        Args:
            scafold: create default experiment structure
        Returns:
            <list> - list of generated configurations of atomic experiment
        """
        return [self.d_cfg]

    def post(self, *args) -> None:
        """
        Post method to generate results gathering and presentation generation
        Args:
            *args:
        Returns:
            <None>
        """
        pass
