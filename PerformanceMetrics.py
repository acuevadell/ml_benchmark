import json
import numpy as np

import matplotlib.pyplot as plt

class PerformanceMetrics:

    def __init__(self):
        self.latencies  = []
        self.file_names = []
        self.file_sizes = []
        self.prediction = []

        self.np_latencies = None
        self.np_filesizes = None

    def add_prediction(self, latency, file_name, file_size, predicted):
        self.latencies.append(latency)
        self.file_names.append(file_name)
        # Bytes to KB
        self.file_sizes.append(file_size / 1024)
        self.prediction.append(predicted)

    def save_predictions(self):
        data = []

        for name, y in zip(self.file_names, self.prediction):
            tmp = {
                'file_name':  name,
                'prediction': y
            }
            data.append(tmp)
        with open("predictions.json", "w") as pfile:
            json.dump(data, pfile)

    def getLatencies(self):
        return {
            "Min"    : np.min(self.np_latencies),
            "P50"    : np.percentile(self.np_latencies, 50),
            "P75"    : np.percentile(self.np_latencies, 75),
            "P90"    : np.percentile(self.np_latencies, 90),
            "P99"    : np.percentile(self.np_latencies, 99),
            "Max"    : np.max(self.np_latencies),
            "Median" : np.median(self.np_latencies),
            "StdDev" : np.std(self.np_latencies),
        }

    def getThroughput(self):
        return {
            "Total Data (MB)"   : np.sum(self.np_filesizes) / 1024,
            "Total Time (s)"    : np.sum(self.np_latencies) / 1000,
            "Throughput (MB/s)" : self.throughput(),
        }

    def throughput(self):
        total_data_mb  = np.sum(self.np_filesizes) / 1024  # KB to MB
        total_time_sec = np.sum(self.np_latencies) / 1000  # ms to sec
        return total_data_mb / total_time_sec if total_time_sec > 0 else 0

    def summary(self):
        return {
            "Throughput" : self.getThroughput(),
            "Latencies"  : self.getLatencies()
        }

    def save_metrics(self):
        if len(self.latencies) != len(self.file_sizes):
            raise ValueError("Latency and data size lists must be the same length.")
        
        self.np_latencies = np.array(self.latencies)
        self.np_filesizes = np.array(self.file_sizes)
        
        summary = self.summary()
        with open("result.json", "w") as pfile:
            json.dump(summary, pfile)

    def save_cdf(self):
        # Sort the data
        sorted_data = np.sort(self.np_latencies)

        # Calculate the CDF values
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

        # Plot the CDF
        plt.plot(sorted_data, cdf)
        plt.xlabel('Value')
        plt.ylabel('CDF')
        #plt.yscale('log')
        plt.title('Cumulative Distribution Function')
        plt.grid(True)
        plt.savefig('cdf.png', bbox_inches='tight')
        #plt.show()
