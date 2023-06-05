import os
import re
import time
import json
import psutil
import pandas as pd
import numpy as np
import subprocess
import threading
from PIL import Image
from paddleocr import PaddleOCR
import Levenshtein as lev

# Constants
BASE_DIR = r'C:\Users\nikla\Desktop\Examensarbete\OCR_Comparison\SciTSR\myTest'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
JSON_DIR = os.path.join(BASE_DIR, 'structure')
PATTERN = re.compile(r'[-_\s]')


class ResourceMonitor:
    """Monitors the CPU and GPU resource usage."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_percentages = []
        self.gpu_percentages = []
        self.stop_monitoring = False
        self.logical_cores = psutil.cpu_count()

    def monitor(self):
        """Monitor CPU and GPU usage in an infinite loop."""
        while not self.stop_monitoring:
            cpu_percent = self.process.cpu_percent(
                interval=self.interval) / self.logical_cores
            self.cpu_percentages.append(cpu_percent)
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu',
                                                  '--format=csv,noheader,nounits'])
                gpu_percent = float(output.decode('utf-8').strip())
            except (subprocess.CalledProcessError, FileNotFoundError):
                gpu_percent = -1
            self.gpu_percentages.append(gpu_percent)

    def start(self):
        """Start monitoring in a separate thread."""
        self.monitoring_thread = threading.Thread(target=self.monitor)
        self.monitoring_thread.start()

    def stop(self):
        """Stop monitoring and calculate the average CPU and GPU usage."""
        self.stop_monitoring = True
        self.monitoring_thread.join()
        avg_cpu_percent = sum(self.cpu_percentages) / len(self.cpu_percentages)
        avg_gpu_percent = sum(self.gpu_percentages) / len(self.gpu_percentages)
        return avg_cpu_percent, avg_gpu_percent


def create_dataframe_and_save(ocr_texts, json_texts, processing_times, sorted_text=False):
    """Creates a pandas dataframe from the provided lists and saves it to an Excel file."""
    DistanceEditDict = {}
    file_name = 'PaddleOCR_Refactored.xlsx' if sorted_text else 'PaddleOCR_Refactored_No_SORT.xlsx'
    for i, filename in enumerate(os.listdir(IMAGE_DIR)):
        DistanceEdit = lev.distance(ocr_texts[i], json_texts[i])
        percentage = (max(len(ocr_texts[i]), len(json_texts[i])) - DistanceEdit) / max(len(
            ocr_texts[i]), len(json_texts[i]))
        filename = os.path.splitext(filename)[0]
        chars_per_millisecond = len(
            ocr_texts[i]) / (processing_times[i] * 1000)
        DistanceEditDict[filename] = {'Distance Edit': DistanceEdit, 'Percentage': percentage,
                                      'CharsPerMillisecond': chars_per_millisecond, 'Time': processing_times[i]}
    df = pd.DataFrame.from_dict(DistanceEditDict, orient='index')
    df.index.name = 'FileName'
    df.to_excel(file_name)


def process_files(image_dir, json_dir):
    """Process image and JSON files from specified directories."""
    ocr = PaddleOCR(lang='en', use_gpu=True, gpu_mem=3000)
    processing_times = []
    ocr_texts = []
    json_texts = []
    sorted_ocr_texts = []
    sorted_json_texts = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            full_filename = os.path.join(image_dir, filename)
            img = Image.open(full_filename).convert('RGB')
            img_array = np.array(img)
            start_time = time.time()
            result = ocr(img_array)
            end_time = time.time()
            processing_times.append(end_time - start_time)
            ocr_text = ' '.join(t[0] for t in result[1]).split()
            sorted_ocr_text = ''.join(sorted(ocr_text))
            sorted_ocr_text = re.sub(PATTERN, '', sorted_ocr_text)
            ocr_texts.append(ocr_text)
            sorted_ocr_texts.append(sorted_ocr_text)

    for json_filename in os.listdir(json_dir):
        full_filename = os.path.join(json_dir, json_filename)
        with open(full_filename, 'r') as f:
            data = json.load(f)
        json_text = []
        for cell in data['cells']:
            json_text.extend(cell['content'])
        sorted_json_text = "".join(sorted(json_text))
        sorted_json_text = re.sub(PATTERN, '', sorted_json_text)
        json_texts.append(json_text)
        sorted_json_texts.append(sorted_json_text)

    return ocr_texts, json_texts, sorted_ocr_texts, sorted_json_texts, processing_times


if __name__ == "__main__":
    resource_monitor = ResourceMonitor(interval=0.5)
    resource_monitor.start()

    start_time = time.time()
    ocr_texts, json_texts, sorted_ocr_texts, sorted_json_texts, processing_times = process_files(
        IMAGE_DIR, JSON_DIR)
    create_dataframe_and_save(ocr_texts, json_texts,
                              processing_times, sorted_text=False)
    create_dataframe_and_save(
        sorted_ocr_texts, sorted_json_texts, processing_times, sorted_text=True)
    end_time = time.time()

    avg_cpu_percent, avg_gpu_percent = resource_monitor.stop()

    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Average CPU usage: {avg_cpu_percent:.2f}%")
    print(f"Average GPU usage: {avg_gpu_percent:.2f}%")
    print(f"RAM usage: {rss:.2f} MB")
