from lib import *

class MetricMonitor:
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"loss": 0, "lr": 0, "count": 0, "avg_loss": 0})

    def update(self, metric_name, loss, lr):
        metric = self.metrics[metric_name]

        metric["loss"] += loss
        metric["Lr"] = lr
        metric["count"] += 1

        metric["avg_loss"] = metric["loss"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg_loss:.{float_precision}f} - LR: {LR:.{float_precision}f}".format(
                    metric_name=metric_name, avg_loss=metric["avg_loss"], LR=metric["Lr"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
    
    def getdic(self):
        return self.metrics.items()[3][1] # value của avg_loss