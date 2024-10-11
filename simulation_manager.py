from dataclasses import dataclass
from datetime import datetime, timedelta

from anomaly_detector import *
from stream_generator import *
from anomaly_generator import *


@dataclass
class Point:
    """
    Represents a single point in the simulation.

    Attributes:
        t (datetime): Timestamp of the point.
        val (float): Value of the point.
        has_anomaly (bool): Whether the point has an anomaly.
        result (DetectionResult): The result of the anomaly detection.
    """

    t: datetime
    val: float
    has_anomaly: bool
    result: DetectionResult


class SimulationManager:
    """
    Manages a simulation of a time series with anomalies.
    """

    def __init__(
        self,
        window_max: int = 2000,
        delta: float = 2,
        every_n_sample: int = 1,
        start_time: datetime = datetime(2001, 1, 1, 0, 0, 0),
        detector: AbstractAnomalyDetector = PEWMAAnomalyDetector(),
        generator: StreamGenerator = SinusoidalPatternGenerator(),
        anomaly_adder: AbstractAnomalyGenerator = RandomizedSpikeAnomalyAdder(),
    ):
        """
        Initializes the simulation manager with parameters for anomaly detection.

        Args:
            window_max (int): Maximum size of the window of points to keep.
            delta (float): Time between each point in hours.
            every_n_sample (int): Sampling period in time steps to call the anomaly detector. This is used to reduce the effect of noise and lower the computational cost.
            start_time (datetime): Starting time of the simulation.
            detector (AbstractAnomalyDetector): The anomaly detector to use.
            generator (StreamGenerator): The stream generator to use.
            anomaly_adder (AbstractAnomalyGenerator): The anomaly generator to use.
        """
        self.window_max = window_max
        self.delta = delta
        self.start_time = start_time
        self.detector = detector
        self.generator = generator
        self.anomaly_adder = anomaly_adder
        self.window: list[Point] = []
        self.every_n_sample = every_n_sample
        self.steps = 0

        self.last_detection = None
        self.t = 0.0

    def step(self, n: int = 1) -> list[Point]:
        """
        Advances the simulation by a specified number of steps.

        Args:
            n (int, optional): Number of steps to advance the simulation. Defaults to 1.

        Returns:
            list[Point]: List of Points representing the new state of the simulation.
        """

        points: list[Point] = []
        for _ in range(n):
            val = self.generator.generate_next(self.delta)
            anomaly_val, flag = self.anomaly_adder.add_anomaly(self.delta)

            val += anomaly_val
            if self.steps % self.every_n_sample == 0:
                result = self.detector.update_reading(val)
                self.last_detection = result

            point = Point(
                self.start_time + timedelta(hours=self.t),
                val,
                flag,
                self.last_detection,
            )
            self.t += self.delta
            self.steps += 1
            points.append(point)

        self.window.extend(points)
        if len(self.window) > self.window_max:
            self.window = self.window[len(self.window) - self.window_max :]

        return self.window
