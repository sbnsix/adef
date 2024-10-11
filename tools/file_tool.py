""" File path manipulation toolset. """

from __future__ import annotations


class FileTool:
    """
    File tool used to help with file operations
    """

    @staticmethod
    def change_path(input_path: str, new_level: str) -> str:
        """
        Method changes last nested directory level to desired one
        in the file name to help organize experiment results.
        Args:
            input_path: file name path to be changed
            new_level: new directory level to be applied
        Returns:
            <str> - new modified file path
        """
        new_path = input_path
        # 1. Normalize path
        new_path = new_path.replace("\\", "/")

        # 2. Find last path element
        idx2 = new_path.rfind("/")
        path_mod = new_path[:idx2]
        idx1 = path_mod.rfind("/") + 1

        # 3. New level normalization - all unnecessary
        #    folder ending characters to avoid any unnecessary
        #    issues with file path after processing completion.
        new_level = new_level.replace("/", "").replace("\\", "")

        # 4. Replace element with new path
        new_path = new_path.replace(new_path[idx1:idx2], new_level)

        return new_path


if "__main__" == __name__:
    test_paths = [
        "../exp/3k_down/5/area1/attack/a_data_e_flat_05.csv",
    ]
    for test_path in test_paths:
        print(f"Test   path: {test_path}")
        result_path = FileTool.change_path(test_path, "output")
        print(f"Result path: {result_path}")
