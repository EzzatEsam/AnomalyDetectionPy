from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


def arr_mean(arr: list[float]) -> float:
    """
    Calculates the mean of an array of numbers.

    Args:
        arr (list[float]): The array of numbers.

    Returns:
        float: The mean of the array.
    """
    return sum(arr) / len(arr)


def arr_std(arr: list[float]) -> float:
    """
    Calculates the standard deviation of an array of numbers.

    Args:
        arr (list[float]): The array of numbers.

    Returns:
        float: The standard deviation of the array.
    """
    mean = arr_mean(arr)
    return math.sqrt(sum([(x - mean) ** 2 for x in arr]) / len(arr))


@dataclass
class DetectionResult:
    """
    Represents the result of anomaly detection.

    Attributes:
        expected_val (float): Expected value of the reading.
        safe_dist (float): Safe distance from the expected value and is equal std * threshold.
        distance (float): The distance from the expected value.
        is_anomaly (bool): Whether the reading is an anomaly.
    """

    expected_val: float
    safe_dist: float
    distance: float
    is_anomaly: bool


class AbstractAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detectors.

    This class provides the interface for anomaly detectors.

    Attributes:
        None

    Methods:
        update_reading(reading: float) -> tuple[float, float]:
            Updates the internal state of the anomaly detector with a new reading.

        is_anomaly(z_score: float) -> bool:
            Checks whether the given z-score is an anomaly.
    """

    @abstractmethod
    def update_reading(self, reading: float) -> DetectionResult:
        """
        Updates the internal state of the anomaly detector with a new reading.

        Args:
            reading (float): The new reading.

        Returns:
            DetectionResult: The result of the anomaly detection.
        """
        raise NotImplementedError

    @abstractmethod
    def is_anomaly(self, z_score: float) -> bool:
        """
        Checks whether the given z-score is an anomaly.

        Args:
            z_score (float): The z-score to check.

        Returns:
            bool: Whether the z-score is an anomaly.
        """
        raise NotImplementedError


class MovingAverageAnomalyDetector(AbstractAnomalyDetector):
    """
    An anomaly detector that uses a moving average to detect anomalies.
    Uses sliding window to calculate the moving average and standard deviation.

    checks if the z-score of the new reading is greater than the threshold.


    pros:
        - Simple to implement and understand
    cons:
        - Needs memory for the window
        - Slow adaptation to new data trends
    """
    def __init__(self, window_size=100, threshold=3.0):
        """
        Initialize the moving average detector.
        :param window_size: Size of the moving window for calculating statistics.
        :param threshold: Z-score threshold for anomaly detection.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.window_vals = []

    def update_reading(self, new_value):
        # Append the new value to the window
        self.window_vals.append(new_value)

        # If the window exceeds the size, remove the oldest value
        if len(self.window_vals) > self.window_size:
            self.window_vals.pop(0)

        # Calculate moving average and standard deviation
        mean = arr_mean(self.window_vals)
        std = arr_std(self.window_vals)

        if std == 0:
            return DetectionResult(mean, 0, 0, False)

        # Calculate the Z-score of the new point
        z_score = (new_value - mean) / std

        safe_dist = std * self.threshold
        return DetectionResult(mean, safe_dist, z_score, abs(z_score) > self.threshold)

    def is_anomaly(self, z_score):
        return abs(z_score) > self.threshold


class ExponentialMovingAverageAnomalyDetector(AbstractAnomalyDetector):
    """
    An implementation of the exponential moving average anomaly detector.
    It's based on the following equation :

        EMA_new = alpha * new_value + (1 - alpha) * EMA_old
    
    pros:
        - Does not require a window
        - More accurate than the simple moving average
        - Simple to implement and understand
    cons:
        - Slow adaptation to new data trends
    
    Refrences:
        https://en.wikipedia.org/wiki/Exponential_smoothing
    """
    def __init__(self, alpha=0.5, threshold=3.0):
        self.alpha = alpha
        self.ema = None
        self.ema_squared = None
        self.threshold = threshold

    def update_reading(self, new_value) -> tuple[float, float]:

        # Initialize EMA and variance on the first point
        if self.ema is None:
            self.ema = new_value
            self.ema_squared = new_value**2
            return DetectionResult(new_value, 0, 0, False)

        # Update EMA for the new point
        self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema

        # Update squared EMA for variance calculation
        self.ema_squared = (
            self.alpha * (new_value**2) + (1 - self.alpha) * self.ema_squared
        )

        # Calculate variance and standard deviation
        variance = self.ema_squared - self.ema**2
        std_dev = (
            math.sqrt(variance) if variance > 0 else 1e-6
        )  # Avoid division by zero

        # Calculate Z-score for the new point
        z_score = abs(new_value - self.ema) / std_dev

        expected = self.ema
        return DetectionResult(
            expected, std_dev * self.threshold, z_score, abs(z_score) > self.threshold
        )

    def is_anomaly(self, z_score):
        """
        Determine if the point is an anomaly based on the Z-score.
        :param z_score: Z-score of the new data point.
        :return: True if the point is an anomaly, False otherwise.
        """
        return abs(z_score) > self.threshold


class PEWMAAnomalyDetector(AbstractAnomalyDetector):
    """
    Implements the probalistic exponential weighted moving average (PEWMA) anomaly detector.
    
   
    pros:
        - Does not require a window
        - Faster adaptation than the simple moving average and exponential moving average  detectors
    cons:
        - Needs steps for training
        - Doesn't take into account the the seasonal nature of the stream
   
    Refrences: 
        Carter, K. M., & Streilein, W. W. (2012). Probabilistic reasoning for streaming anomaly detection. 2012 IEEE Statistical Signal Processing Workshop (SSP), 377-380. doi:10.1109/ssp.2012.6319708
    """
    def __init__(self, alpha=0.97, beta=0.99, threshold=3.0, training_steps=50):
        """
        Initialize the PEWMA parameters.

        Args:
            alpha (float): Weight for the moving average (0 < alpha < 1).
            beta (float): Weight for the probability adjustment (0 < beta < 1).
            threshold (float): Z-score threshold for anomaly detection.
        """
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.training_steps = training_steps

        self.steps = 0

        self.std = None
        self.mean = None
        self.s1 = None
        self.s2 = None

    def update_reading(self, reading: float):
        """
        Update the PEWMA with a new reading and calculate the Z-score.

        Args:
            reading (float): The new reading.

        Returns:
            tuple[float, float]: The current mean and Z-score.
        """
        self.steps += 1

        # initialize parameters
        if self.s1 is None:
            self.s1 = reading
        if self.s2 is None:
            self.s2 = reading**2
        if self.mean is None:
            self.mean = reading

        self.std = math.sqrt(self.s2 - self.s1**2)

        if self.std == 0:
            zt = 0
        else:
            zt = (reading - self.mean) / self.std

        pt = math.exp(-(zt**2) / 2) / math.sqrt(2 * math.pi)

        if self.steps < self.training_steps:
            alpha_t = 1 - 1 / self.steps
        else:
            alpha_t = (1 - self.beta * pt) * self.alpha

        self.s1 = alpha_t * self.s1 + (1 - alpha_t) * reading
        self.s2 = alpha_t * self.s2 + (1 - alpha_t) * reading**2

        self.mean = self.s1

        safe_dist = self.std * self.threshold

        return (
            DetectionResult(self.mean, safe_dist, zt, self.is_anomaly(zt))
            if self.steps >= self.training_steps
            else DetectionResult(self.mean, safe_dist, zt, False)
        )

    def is_anomaly(self, z_score: float) -> bool:
        """
        Check if the given Z-score indicates an anomaly.

        Args:
            z_score (float): The Z-score to check.

        Returns:
            bool: True if the Z-score indicates an anomaly, otherwise False.
        """
        return abs(z_score) > self.threshold
