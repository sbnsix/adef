
from __future__ import annotations
import os
import sys
import site
import shutil
import glob


class Patcher:
    """
    Class implementing automatic file patcher that replaces files in python site-packages
    folder with ones found under patch directory.
    """
    @staticmethod
    def run() -> None:
        """
        Method implements Python library patching for libraries needed by ADEF
        Args:
            <None>
        Returns:
            <None>
        """
        print(f"Starting ADEF Python library patch process")

        curr_path = os.getcwd()
        spl_path = ""
        for site_lib_path in site.getsitepackages():
            if "site-packages" in site_lib_path:
                spl_path = site_lib_path
                break

        # Find all Python patch files
        patch_folders = [y.replace(curr_path, "") for x in os.walk(os.getcwd()) for y in glob.glob(os.path.join(x[0], "*.py"))]

        # Skip self
        patch_folders = [ x for x in patch_folders if not x.endswith("/patch.py") and not x.endswith("\patch.py") ]

        patch_files_ok_cnt = 0
        patch_files_fail_cnt = 0

        # Patch all files in any required library
        for file in patch_folders:
            # Prepare path for file copy
            src_path = file
            if src_path.startswith("\\") or src_path.startswith('/'):
                src_path = src_path[1:]

            dst_path = f"{spl_path}{file}"

            # To guarantee successful operation check src and dst existence
            if os.path.isfile(src_path) and os.path.isfile(dst_path):
                print(f"Patching file: |{src_path}| to |{dst_path}|")
                try:
                    shutil.copyfile(src_path, dst_path)
                    patch_files_ok_cnt += 1
                except OSError as ex:
                    print(ex)
                    patch_files_fail_cnt += 1

        print(f"Completed ADEF Python library patch process with:")
        print(f"\tSuccessful  : {patch_files_ok_cnt} file{'s' if patch_files_ok_cnt > 1 else ''}")
        print(f"\tUnsuccessful: {patch_files_fail_cnt} file{'s' if patch_files_fail_cnt > 1 else ''}")


if __name__ == "__main__":
    Patcher.run()
