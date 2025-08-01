import numpy as np

class PerformanceMetrics:
    def __init__(self, latencies_ms, data_sizes_kb):
        if len(latencies_ms) != len(data_sizes_kb):
            raise ValueError("Latency and data size lists must be the same length.")
        self.latencies = np.array(latencies_ms)
        self.data_sizes = np.array(data_sizes_kb)

    def percentiles(self):
        return {
            "P50": np.percentile(self.latencies, 50),
            "P90": np.percentile(self.latencies, 90),
            "P99": np.percentile(self.latencies, 99)
        }

    def basic_stats(self):
        return {
            "Mean": np.mean(self.latencies),
            "Median": np.median(self.latencies),
            "StdDev": np.std(self.latencies),
            "Min": np.min(self.latencies),
            "Max": np.max(self.latencies)
        }

    def throughput(self):
        total_data_mb = np.sum(self.data_sizes) / 1024  # KB to MB
        total_time_sec = np.sum(self.latencies) / 1000  # ms to sec
        return total_data_mb / total_time_sec if total_time_sec > 0 else 0

    def summary(self):
        return {
            "Percentiles (ms)": self.percentiles(),
            "Basic Stats (ms)": self.basic_stats(),
            "Total Data (MB)": np.sum(self.data_sizes) / 1024,
            "Total Time (s)": np.sum(self.latencies) / 1000,
            "Throughput (MB/s)": self.throughput()
        }

    def print_summary(self):
        summary = self.summary()
        for key, value in summary.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval:.2f}")
            else:
                print(f"  {value:.2f}")
