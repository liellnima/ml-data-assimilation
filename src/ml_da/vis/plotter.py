from typing import Literal

import matplotlib.pyplot as plt


def plot_metrics(results, model_name=Literal["4D-Var", "EnKF", "Neural EnKF"]):
    """
    model_name: name of the model used. Plot one method
    """
    metrics = results["metrics"]  # Idk what our method will return so in case we return more than just the metrics.

    for metric_name, values in metrics.items():
        if len(values) == 0:
            continue

        plt.figure()
        plt.plot(values, label=model_name)

        plt.title(f"{metric_name.upper()} over time")
        plt.xlabel("Time")
        plt.ylabel(metric_name)  # may need plt.yscale("log") for some metrics if looks weird.
        plt.legend()
        plt.show()


def compare_models(results_dict):
    """
    If we want to plot multiple methods for each metrics graph results_dict = { "EnKF": results_enkf, "Var-4D":

    results_var, "NeuralEnKF": results_neural, }
    """

    # Get all metric names
    metric_names = list(next(iter(results_dict.values()))["metrics"].keys())

    for metric in metric_names:
        plt.figure()

        for model_name, results in results_dict.items():
            values = results["metrics"][metric]

            if len(values) > 0:
                plt.plot(values, label=model_name)

        plt.title(f"{metric.upper()} comparison")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.legend()
        plt.show()
