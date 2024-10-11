from __future__ import annotations

import time
import pandas as pd


class PIDController:
    def __init__(self):
        self.previous_error = 0
        self.integral = 0

    def run(self, setpoint: float) -> pd.DataFrame:
        """
        Method runs simple Proportional Integration Derivative controller model
        Args:
            setpoint: Initial setpoint condition
        Returns:
            <pd.DataFrame> - dataframe combining model output data
        """
        while True:
            error = setpoint - measured_value
            proportional = error
            self.integral = self.integral + error * dt
            derivative = (error - self.previous_error) / dt
            output = Kp * proportional + Ki * self.integral + Kd * derivative
            previous_error = error
            # time.sleep(dt)
