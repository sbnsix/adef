""" POSIX time manipulation toolset. """

from __future__ import annotations

import datetime
import re

import pytz


class Posix:
    """
    Class Posix operation
    """

    @staticmethod
    def epoch(tz_info: pytz.tzinfo = pytz.utc) -> datetime.datetime:
        """
        Method returns epoch time in specific timezone
        Args:
            tz_info - time zone information about correct time zone to be used
        Returns:
            <datetime> - Datetime.datetime object(Jan 1st 1970)
        """
        epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
        epoch = epoch.replace(tzinfo=tz_info)
        return epoch

    @staticmethod
    def time(dt: datetime.datetime, tz_info: pytz.tzinfo = pytz.utc) -> int:
        """
        Converts Python datetime to unix timestamp
        Args:
            dt - Python datetime.datetime time to be converted
            tz_info - time zone information about correct time zone to be used
        Returns:
            <int> - POSIX time expressed in seconds
        """
        if dt.tzname() is None:
            dtn = dt.replace(tzinfo=tz_info)
        else:
            dtn = dt.astimezone(tz_info).replace(tzinfo=tz_info)
        return (dtn - Posix.epoch(tz_info)).total_seconds()

    @staticmethod
    def time_ms(dt: datetime.datetime, tz_info: pytz.tzinfo = pytz.utc) -> float:
        """
        Args:
            dt - Python datetime.datetime time to be converted
            tz_info - time zone information about correct time zone to be used
        Returns:
            <float> - POSIX time expressed in milliseconds
        """
        return round(float(Posix.time(dt, tz_info) * 1000.0), 3)

    @staticmethod
    def time_wait(dt: datetime.datetime, cycle_time: int = 15) -> int:
        """
        Method comuptes wait time expressed in seconds adjusted to given time
        within an hour
        Args:
            dt          - Python datetime.datetime time to be converted
            cycle_time  - cycle time expressed in seconds within an hour
        Returns:
            <int>   - time in seconds to next wake-up time
        """
        # Convert time to number of 15 min periods
        current_seconds = (dt.minute * 60) + dt.second
        while (cycle_time - current_seconds) < 0:
            current_seconds -= cycle_time

        # Normalize time
        current_seconds = (
            cycle_time
            if (0 == (cycle_time - current_seconds))
            else (cycle_time - current_seconds)
        )

        return current_seconds

    @staticmethod
    def time_dt(time_seconds: int) -> datetime.datetime:
        """
        Args:
            time        - integer value expressing number of seconds from Jan 1st 1970.
        Returns:
            <datetime>  - Python datetime time object
        """
        if int is type(time_seconds):
            return Posix.epoch() + datetime.timedelta(seconds=time_seconds)
        raise ValueError(f"Incorrect input type: {str(type(time_seconds))}")

    @staticmethod
    def align(dt: datetime.datetime, resolution: int) -> int:
        """
        Method aligns given time to specific time resolution expressed in seconds
        Args:
            dt          - Python datetime.datetime time to be converted
            resolution  - Amount of time used to align timestamp
        Returns:
            <int> - aligned POSIX time expressed in milliseconds
        """
        time_ms = float(Posix.time(dt))
        time_aligned = time_ms - (time_ms % resolution)
        return time_aligned

    @staticmethod
    def c2cas_time(tm: time.time, tz_info: pytz.tzinfo = pytz.utc) -> int:
        """
        Method converts time in any expected form to Cassandra DB timestamp (long) format
        Args:
            tm      - time that is requried to be converted
        Returns:
            <int>   - converted time in form of seconds
        """
        if isinstance(tm, int):
            return tm
        if isinstance(tm, datetime.datetime):
            tm = int(Posix.time_ms(tm, tz_info))

        if isinstance(tm, str):
            date_fmt = "%Y-%m-%d"
            time_fmt = "%H:%M:%S"
            date_pat = r"[\d]{4}-[\d]{2}-[\d]{2}"
            time_pat = r"[\d]{2}:[\d]{2}:[\d]{2}"
            time_patterns = {
                date_fmt + "/" + time_fmt: re.compile(date_pat + "/" + time_pat),
                date_fmt + " " + time_fmt: re.compile(date_pat + " " + time_pat),
            }
            for m, p in time_patterns.items():
                if p.match(tm) is not None:
                    # Fix timeshifting bug - strptime doesn't contain time zone information
                    # return int(Posix.time_ms(parse_dt(tm, tzinfos={"UTC": -100})))
                    # datetime.datetime.strptime(tm, m))) #+'-'+tz, m+'-%Z')))
                    return int(
                        Posix.time_ms(datetime.datetime.strptime(tm, m), tz_info)
                    )  # +'-'+tz, m+'-%Z')))
        return tm
