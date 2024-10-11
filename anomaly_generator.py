from abc import ABC, abstractmethod
import random


class AbstractAnomalyGenerator(ABC):
    """
    Abstract base class for anomaly generators.

    Attributes:
        None

    Methods:
        add_anomaly(delta: float) -> tuple[float, bool]:
            Adds an anomaly to the stream.

            Args:
                delta (float): Time since the last generated value.

            Returns:
                tuple[float, bool]: A tuple containing the next value of the stream and a boolean indicating whether an anomaly was added.
    """
    @abstractmethod
    def add_anomaly(self, delta: float) -> tuple[float, bool]:
        pass



class RandomizedSpikeAnomalyAdder(AbstractAnomalyGenerator):
    """ 
    Adds random spikes with random durations and amplitudes to the stream at random times.
    """
    def __init__(
        self,
        time_between_anomalies_min: int = 5 * 24,
        time_between_anomalies_max: int = 80 * 24,
        spike_duration_min: int = 6,
        spike_duration_max: int = 2 * 24,
        spike_amplitude_min: float = 0.8,
        spike_amplitude_max: float = 2.6,
) -> None:
        """
        Initializes the simulation class with parameters for anomaly generation
        and spike behavior.

        Args:
            time_between_anomalies_min (int): Minimum time between anomalies in hours (default: 2 days).
            time_between_anomalies_max (int): Maximum time between anomalies in hours (default: 80 days).
            spike_duration_min (int): Minimum duration of a spike in hours (default: 6 hours).
            spike_duration_max (int): Maximum duration of a spike in hours (default: 2 days).
            spike_amplitude_min (float): Minimum amplitude of the spike (default: 0.8).
            spike_amplitude_max (float): Maximum amplitude of the spike (default: 2.6).
        """
        super().__init__()

        self.time_between_anomalies_min = time_between_anomalies_min
        self.time_between_anomalies_max = time_between_anomalies_max
        self.spike_duration_min = spike_duration_min
        self.spike_duration_max = spike_duration_max
        self.spike_amplitude_min = spike_amplitude_min
        self.spike_amplitude_max = spike_amplitude_max

        self.current_spike_duration = 0
        self.current_spike_amp = 0
        self.t = 0.0
        self.is_applying = False
        self.last_anomaly_time = 0.0
        self.next_anomaly_time = self._get_random_anomaly_time()

    def _get_random_anomaly_time(self) -> float:
        """
        Generates a random time between the minimum and maximum times between anomalies
        as set in the constructor.

        Returns:
            float: A random time between the minimum and maximum times between anomalies in hours.
        """
        return random.uniform(
            self.time_between_anomalies_min, self.time_between_anomalies_max
        )

    def _get_random_spike_duration(self) -> float:
        """
        Generates a random duration for a spike between the minimum and maximum
        durations set in the constructor.

        Returns:
            float: A random duration between the minimum and maximum spike durations in hours.
        """

        return random.uniform(self.spike_duration_min, self.spike_duration_max)

    def _get_random_spike_amplitude(self) -> float:
        """
        Generates a random amplitude for a spike between the minimum and maximum
        amplitudes set in the constructor.

        Returns:
            float: A random amplitude between the minimum and maximum spike amplitudes.
        """

        return random.uniform(
            self.spike_amplitude_min, self.spike_amplitude_max
        ) * random.choice([-1, 1])

    def _get_current_spike_amp(self) -> float:
        """
        Calculates the current amplitude of the spike based on the time elapsed since the
        start of the spike.

        The amplitude is calculated by linearly interpolating between the maximum amplitude
        and zero over the duration of the spike.

        This equation is based on the triangle wave function
        
        Returns:
            float: The current amplitude of the spike.
        """
        anomaly_time = self.t - self.last_anomaly_time
        if anomaly_time <= 0.5 * self.current_spike_duration:
            return self.current_spike_amp * (
                anomaly_time / (0.5 * self.current_spike_duration)
            )
        else:
            return self.current_spike_amp * (
                (self.current_spike_duration - anomaly_time)
                / (0.5 * self.current_spike_duration)
            )

    def add_anomaly(self, delta) -> tuple[float, bool]:
        """
        Advances the anomaly simulator by delta hours.

        If we are in the middle of an anomaly, it will return the current amplitude of the anomaly
        and True. If we are not in the middle of an anomaly and the time since the last anomaly is
        greater than or equal to the time until the next anomaly, it will start a new anomaly and
        return 0 and False. Otherwise, it will return 0 and False.

        Args:
            delta (float): Time in hours to advance the simulator.

        Returns:
            tuple[float, bool]: The amplitude of the current anomaly and whether we are in the
            middle of an anomaly.
        """
        self.t += delta

        if self.is_applying:
            if self.t - self.last_anomaly_time >= self.current_spike_duration:
                self.is_applying = False
                self.last_anomaly_time = self.t
                return 0, False
            else:
                return self._get_current_spike_amp(), True

        elif self.t - self.last_anomaly_time >= self.next_anomaly_time:
            self.next_anomaly_time = self._get_random_anomaly_time()
            self.current_spike_duration = self._get_random_spike_duration()
            self.current_spike_amp = self._get_random_spike_amplitude()
            self.is_applying = True
            self.last_anomaly_time = self.t
            return 0, False
        else:
            return 0, False
