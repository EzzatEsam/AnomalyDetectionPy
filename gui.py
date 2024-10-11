import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from anomaly_detector import (
    ExponentialMovingAverageAnomalyDetector,
    MovingAverageAnomalyDetector,
    PEWMAAnomalyDetector,
)
from simulation_manager import SimulationManager


class SimulationGUI:
    """
    A tkinter GUI for the simulation manager. 
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simulation Manager")
        self.simulation_manager: SimulationManager = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.call_id = None

        # Store points
        self.points = []
        self.create_gui()

    def on_closing(self):
        exit()

    def create_gui(self):
        # Create matplotlib figure and axes
        """
        Creates the GUI elements for the simulation manager.

        This function creates a matplotlib figure with three subplots and a canvas
        for the figure. The canvas is placed in the root window with a grid layout.
        The function also creates a dropdown (combobox) for selecting the anomaly
        detection algorithm, a label for the dropdown, and a button to start the
        simulation. The elements are arranged in a grid layout with the canvas
        taking up most of the space and the other elements aligned to the right.
        """
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(10, 6), sharex=False, layout="constrained"
        )

        self.ax1_side = self.ax1.twinx()

        # Create a canvas for the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        # Use grid to arrange elements
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=6, sticky="nsew")

        # Configure grid weights for resizing
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        self.label_info = ttk.Label(
            self.root,
            text="Select the algorithm you want to use. The default option is PEWMA.\n\n"
            "1. SMA - Simple Moving Average (sliding window)\n\n"
            "2. EMA - Exponential Moving Average \n\n"
            "3. PEWMA - Probability Weighted Exponential Moving Average",
            justify="left",
            wraplength=200,
        )

        self.label_info.grid(row=0, column=1, padx=15, pady=15, sticky="w")
        self.option_var = tk.StringVar(self.root, value="PEWMA")
        self.dropdown = ttk.Combobox(
            self.root,
            textvariable=self.option_var,
            values=["SMA", "EMA", "PEWMA"],
        )
        self.dropdown.grid(row=1, column=1, padx=15, pady=5, sticky="w")

        # Start button below the fields
        self.start_button = ttk.Button(
            self.root, text="Start Simulation", command=self.start_simulation
        )
        self.start_button.grid(row=2, column=1, padx=15, pady=15, sticky="w")

        # Center the right panel vertically by adding padding and adjusting row weights
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

    def start_simulation(self):
        """
        Starts the simulation with the selected anomaly detector.

        Cancels any existing simulation and schedules the first update of the
        graph using the selected anomaly detector. The selected detector is
        created with default parameters.

        """
        if self.call_id is not None:
            self.root.after_cancel(self.call_id)
            self.call_id = None

        match self.option_var.get():
            case "SMA":
                detector = detector = MovingAverageAnomalyDetector(window_size=150)

            case "EMA":
                detector = ExponentialMovingAverageAnomalyDetector(
                    alpha=0.03, threshold=2.5
                )
            case "PEWMA":
                detector = PEWMAAnomalyDetector(
                    alpha=0.95, beta=0.06, threshold=3.2, training_steps=100
                )

            case _:
                raise ValueError("Invalid option")

        self.simulation_manager = SimulationManager(
            window_max=7000, delta=0.5, every_n_sample=4, detector=detector
        )
        self.update_graph()

    def update_graph(self):
        # Get new data points from the simulation
        """
        Updates the graph with new data points from the simulation.

        This function is called recursively to update the graph at regular
        intervals. It clears the old plots, extracts the new data from the
        simulation, and plots the new data. The graph is then redrawn and the
        function schedules itself to be called again after 1 second.

        """
        new_points = self.simulation_manager.step(250)
        self.points = new_points

        # Clear old plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax1_side.clear()

        # Extract data
        times = [point.t for point in self.points]
        values = [point.val for point in self.points]
        truth = [point.has_anomaly for point in self.points]
        z_scores = [abs(point.result.distance) for point in self.points]
        is_anomalies = [1 if point.result.is_anomaly else 0 for point in self.points]
        upper_bounds = [
            point.result.expected_val + point.result.safe_dist for point in self.points
        ]
        lower_bounds = [
            point.result.expected_val - point.result.safe_dist for point in self.points
        ]

        # Plot Value over Time
        self.ax1.set_title("Data stream")
        self.ax1.set_ylabel("Value", color="blue")
        self.ax1.tick_params(axis="y", labelcolor="blue")
        self.ax1.plot(times, values, label="signal")
        self.ax1.plot(
            times,
            [point.result.expected_val for point in self.points],
            label="Expected Value",
            color="green",
            linestyle="dashed",
        )
        self.ax1.fill_between(
            times,
            lower_bounds,
            upper_bounds,
            color="yellow",
            alpha=0.2,
            label="Safe Distance",
        )
        self.ax1.legend(loc = "upper left")

        self.ax1_side.plot(
            times, truth, label="Value", color="red", linestyle="dotted"
        )
        self.ax1_side.set_ylabel("Anomaly (Truth)", color="red")
        self.ax1_side.set_ylim(0, 1)
        self.ax1_side.yaxis.set_label_position("right")
        self.ax1_side.tick_params(axis="y", labelcolor="red")

        # Plot Z-Score over Time
        self.ax2.set_title("Z-Score")
        self.ax2.plot(times, z_scores, color="blue", label="Z-Score")
        self.ax2.set_ylabel("Absolute Z-Score")
        self.ax2.set_ylim(0, 7)
        self.ax2.grid()
        self.ax2.legend()

        # Plot Anomaly Detection over Time
        self.ax3.set_title("Anomaly Detection")
        self.ax3.plot(times, is_anomalies, color="red", label="Anomaly (Detected)")
        self.ax3.set_ylabel("Anomaly")
        self.ax3.set_xlabel("Time")
        self.ax3.set_ylim(0, 1)
        self.ax3.legend(loc = "upper right")

        # Redraw the canvas
        self.canvas.draw()
        # Schedule the next update after 100 ms
        self.call_id = self.root.after(100, self.update_graph)  # store call id
