import time
import psutil
import logging
from tensorflow.keras.callbacks import Callback

class CpuTemperatureMonitor(Callback):
    def __init__(self, max_temp=75.0, check_interval=60):
        super().__init__()
        self.max_temp = max_temp
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

    def on_train_batch_begin(self, batch, logs=None):
        temp = self.get_cpu_temperature()
        while temp is not None and temp > self.max_temp:
            self.logger.warning(
                f"⚠️ CPU temperature is {temp:.1f}°C, waiting for it to cool down below {self.max_temp}°C..."
            )
            time.sleep(self.check_interval)
            temp = self.get_cpu_temperature()

    def get_cpu_temperature(self):
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            return temps["coretemp"][0].current
        elif temps:
            return list(temps.values())[0][0].current
        else:
            self.logger.warning("⚠️ No CPU temperature sensor found.")
            return None

