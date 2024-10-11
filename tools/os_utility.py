""" OS utility module used to handle OS related operations """


import os
import platform
from datetime import datetime

import psutil


class Utility:
    @staticmethod
    def auto_shutdown(test_time: datetime, log: object) -> bool:
        """
        Method automatically shuts down machine
        after simulation completion.
        Args:
            test_time: shutdown time
            log: logger object
        Returns:
            <bool>  - True if shutdown will be performed
                      False otherwise
        """
        # Detect computer activity and if it is inactive shut it down
        if not Utility.activity_check(test_time):
            os_name = platform.system().lower()
            # Perform system shutdown
            if "windows" in os_name:
                log.warn("Shutting down the machine.")
                os.system("shutdown /s /t 10")
                return True
            elif "linux" in os_name:
                log.warn("Shutting down the machine.")
                os.system("shutdown -h now")
                return True
            else:
                print(f"Unknown OS name {os_name}")

        return False

    @staticmethod
    def if_process_is_running_by_name(process_name: str = "chrome.exe") -> bool:
        """
        Method checks if given process is running.
        Args:
            process_name - name of the processs
        Returns:
            <bool>  - True if process is still running
                      False otherwise
        """
        for proc in psutil.process_iter(["pid", "name"]):
            # This will check if there exists any process running with executable name
            if proc.info["name"] == process_name:
                return True

        return False

    @staticmethod
    def activity_check(test_time: datetime = datetime.now()) -> bool:
        """
        Method check if computer is active returns activity check result.
        Args:
            test_time -
        Returns:
            <bool>  - True if activity is ongoing, False if there is no activty observed
        """
        check_result = False

        # If time is between 8:00am and 11:00pm do not shut down
        if test_time.hour > 7 and test_time.hour < 23:
            check_result = True

        # Activity check - if browser is enabled
        else:
            chrome = Utility.if_process_is_running_by_name("chrome.exe")
            firefox = Utility.if_process_is_running_by_name("firefox.exe")
            if chrome or firefox:
                check_result = True

        return check_result


def main():
    """
    Test method
    """
    times = [
        datetime(year=2021, month=6, day=12, hour=6, minute=15),
        datetime(year=2021, month=6, day=12, hour=7, minute=59),
        datetime(year=2021, month=6, day=12, hour=8, minute=00),
        datetime(year=2021, month=6, day=12, hour=22, minute=59),
        datetime(year=2021, month=6, day=12, hour=23, minute=00),
    ]
    for time in times:
        # result = utility.activity_check(time)
        result = Utility.auto_shutdown(time)
        print(f"T|{str(time)}| => {str(result)}")


if __name__ == "__main__":
    main()
