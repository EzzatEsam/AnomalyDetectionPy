from abc import ABC, abstractmethod
import math
import random


class StreamGenerator(ABC):
    """
    Abstract base class for stream generators.
    """

    @abstractmethod
    def generate_next(self, delta: float) -> float:
        """
        Generates the next value of the stream.

        Args:
            delta (float): Time since the last generated value.

        Returns:
            float: The next value of the stream.

        Raises:
            NotImplementedError: This method must be implemented by subclass.
        """
        raise NotImplementedError


class SinusoidalPatternGenerator(StreamGenerator):
    """
    Generates a sinusoidal pattern. has both seasonal and weekly trends. Has a random noise component.
    """

    def __init__(
        self,
        seasonal_freq: float = 1 / (6 * 30 * 24),
        seasonal_amp: float = 1.0,
        seasonal_sin_ratio: float = 1.0,
        seasonal_cos_ratio: float = 0.5,
        weekly_freq: float = 1 / (20 * 24),
        weekly_amp: float = 0.5,
        weekly_sin_ratio: float = 1.0,
        weekly_cos_ratio: float = 0.3,
        noise_std: float = 0.04,
        offset: float = 2.0,
    ) -> None:
        """
        Initializes the simulation class with parameters for seasonal and weekly
        trends, as well as noise and offset.

        Args:
            seasonal_freq (float): Frequency of the seasonal component in cycles per hour (default: 1/(6 * 30 * 24)).
            seasonal_amp (float): Amplitude of the seasonal component (default: 1.0).
            seasonal_sin_ratio (float): Sine ratio for the seasonal component (default: 1.0).
            seasonal_cos_ratio (float): Cosine ratio for the seasonal component (default: 0.5).
            weekly_freq (float): Frequency of the weekly component in cycles per hour (default: 1/(20 * 24)).
            weekly_amp (float): Amplitude of the weekly component (default: 0.5).
            weekly_sin_ratio (float): Sine ratio for the weekly component (default: 1.0).
            weekly_cos_ratio (float): Cosine ratio for the weekly component (default: 0.3).
            noise_std (float): Standard deviation of the noise added to the signal (default: 0.04).
            offset (float): Offset added to the final signal (default: 2.0).
        """
        super().__init__()

        self.seasonal_freq = seasonal_freq
        self.seasonal_amp = seasonal_amp
        self.seasonal_sin_ratio = seasonal_sin_ratio
        self.seasonal_cos_ratio = seasonal_cos_ratio

        self.weekly_freq = weekly_freq
        self.weekly_amp = weekly_amp
        self.weekly_sin_ratio = weekly_sin_ratio
        self.weekly_cos_ratio = weekly_cos_ratio

        self.noise_std = noise_std
        self.offset = offset

        self.t = 0.0

    def generate_next(self, delta: float) -> float:
        """
        Generates the next value in the simulation.

        Args:
            delta (float): Time since the last generated value.

        Returns:
            float: The next value in the simulation.
        """
        self.t += delta
        wave = (
            self._get_seasonal_delta() + self._get_weekly_delta()
        )  # add seasonal and weekly components
        wave += self._generate_noise()  # add noise
        wave += self.offset  # add offset
        return wave

    def _get_seasonal_delta(self) -> float:
        """
        Calculates the seasonal component of the signal using two sinousoidal waves added together.

        Returns:
            float: The calculated seasonal component.
        """
        return self.seasonal_amp * self.seasonal_sin_ratio * math.sin(
            2 * math.pi * self.t * self.seasonal_freq
        ) + self.seasonal_amp * self.seasonal_cos_ratio * math.cos(
            3 * 2 * math.pi * self.t * self.seasonal_freq
        )

    def _get_weekly_delta(self) -> float:
        """
        Calculates the weekly component of the signal using two sinousoidal waves added together.

        Returns:
            float: The calculated weekly component.
        """
        return self.weekly_amp * self.weekly_sin_ratio * math.sin(
            2 * math.pi * self.t * self.weekly_freq
        ) + self.weekly_amp * self.weekly_cos_ratio * math.cos(
            3 * 2 * math.pi * self.t * self.weekly_freq
        )

    def _generate_noise(self):
        """
        Generates a random noise value from a normal distribution with a standard deviation of self.noise_std and a mean of 0.

        Returns:
            float: The generated noise value.
        """
        return random.normalvariate(0, self.noise_std)
