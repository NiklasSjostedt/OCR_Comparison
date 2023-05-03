import easyocr
import os
import Levenshtein as lev
from PIL import Image
import json
import pandas as pd
import time
import numpy as np
import re
import subprocess
import psutil
from threading import Thread


def get_process_info():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB

    return rss


class CPUMonitor:
    def __init__(self, interval=0.5):
        # Time interval (in seconds) for updating the CPU and GPU usage data
        self.interval = interval
        self.process = psutil.Process(os.getpid())  # Get the current process
        self.cpu_percentages = []  # List to store the CPU usage percentages
        self.gpu_percentages = []  # List to store the GPU usage percentages
        self.stop_monitoring = False  # Flag to stop the monitoring loop
        # Get the total number of logical cores in the system
        self.logical_cores = psutil.cpu_count()

    def monitor(self):
        # Monitoring loop
        while not self.stop_monitoring:
            # Get the CPU usage percentage for the current process and normalize it by the number of logical cores
            cpu_percent = self.process.cpu_percent(
                interval=self.interval) / self.logical_cores
            # Add the CPU usage percentage to the list
            self.cpu_percentages.append(cpu_percent)

            # Get the GPU usage percentage using the nvidia-smi command
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                gpu_percent = float(output.decode('utf-8').strip())
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If nvidia-smi command fails or is not available, set GPU usage to -1
                gpu_percent = -1
            # Add the GPU usage percentage to the list
            self.gpu_percentages.append(gpu_percent)

    def start(self):
        # Start the monitoring loop in a separate thread
        self.monitoring_thread = Thread(target=self.monitor)
        self.monitoring_thread.start()

    def stop(self):
        # Stop the monitoring loop and calculate the average CPU and GPU usage
        self.stop_monitoring = True
        self.monitoring_thread.join()
        avg_cpu_percent = sum(self.cpu_percentages) / len(self.cpu_percentages)
        avg_gpu_percent = sum(self.gpu_percentages) / len(self.gpu_percentages)
        return avg_cpu_percent, avg_gpu_percent


def main():

    total_processing_time = []
    total_start_time = time.time()

    reader = easyocr.Reader(['en'], gpu=True)
    directory = r'C:\Users\nikla\Desktop\Examensarbete\OCR_Comparison\SciTSR\myTest\images'
    json_directory = r"C:\Users\nikla\Desktop\Examensarbete\OCR_Comparison\SciTSR\myTest\structure"

    ocr_complete_sort = []
    ocr_texts = []
    processing_times = []

    for filename in os.listdir(directory):
        filename = directory + "\\" + filename
        if filename.endswith('.png'):
            img = Image.open(filename).convert('RGB')
            img_array = np.array(img)
            start_time = time.time()
            result = reader.readtext(img_array)
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            ocrtext = [t[1] for t in result]
            ocr_texts_formatted = []
            for word in ocrtext:
                # Split the word on underscores and dashes
                split_words = re.split('_|- ', word)
                # Remove underscores from the split words
                split_words = [w for w in split_words if w != '']
                # new_array += split_words)
                ocr_texts_formatted += split_words
            ocrtext_join = ' '.join(ocr_texts_formatted)
            ocrtext_split = ocrtext_join.split()
            ocrtext_sort = sorted(ocrtext_split)
            ocrtext_joinagain = ''.join(ocrtext_sort).replace('_', '')

            ocr_complete_sort.append("".join(sorted(ocrtext_join)).replace(
                ' ', '').replace('-', '').replace('_', ''))
            ocr_texts.append(ocrtext_joinagain.replace(
                ' ', '').replace('-', '').replace('_', ''))

    json_complete_sort = []
    json_texts = []

    for jsonfilename in os.listdir(json_directory):
        jsonfilename = json_directory + '\\' + jsonfilename
        with open(jsonfilename, 'r') as f:
            data = json.load(f)
        json_text = []
        for cell in data['cells']:
            json_text.extend(cell['content'])
        contentSortedList = sorted(json_text)
        contentSortedString = "".join(contentSortedList)
        json_complete_sort.append("".join(sorted(contentSortedString)).replace(
            ' ', '').replace('-', '').replace('_', ''))
        json_texts.append(contentSortedString.replace(
            ' ', '').replace('-', '').replace('_', ''))

    # print(json_complete_sort)
    DistanceEditDict = {}
    for i in range(len(ocr_texts)):
        DistanceEdit = lev.distance(ocr_texts[i], json_texts[i])
        percentage = (max(len(ocr_texts[i]), len(json_texts[i])) -
                      DistanceEdit)/max(len(ocr_texts[i]), len(json_texts[i]))
        filename = os.path.splitext(
            os.path.basename(os.listdir(directory)[i]))[0]
        chars_per_millisecond = len(
            ocr_texts[i]) / (processing_times[i] * 1000)
        DistanceEditDict[filename] = {'Distance Edit': DistanceEdit,
                                      'Percentage': percentage, 'CharsPerMillisecond': chars_per_millisecond, 'Time': processing_times[i]}

    df = pd.DataFrame.from_dict(DistanceEditDict, orient='index')
    df.index.name = 'FileName'
    df.to_excel('EasyOCR_NoSort_GPU.xlsx')

    for i in range(len(ocr_complete_sort)):
        DistanceEdit = lev.distance(
            ocr_complete_sort[i], json_complete_sort[i])
        percentage = (max(len(ocr_complete_sort[i]), len(json_complete_sort[i])) -
                      DistanceEdit)/max(len(ocr_complete_sort[i]), len(json_complete_sort[i]))
        filename = os.path.splitext(
            os.path.basename(os.listdir(directory)[i]))[0]
        chars_per_millisecond = len(
            ocr_complete_sort[i]) / (processing_times[i] * 1000)
        DistanceEditDict[filename] = {'Distance Edit': DistanceEdit,
                                      'Percentage': percentage, 'CharsPerMillisecond': chars_per_millisecond, 'Time': processing_times[i]}

    df = pd.DataFrame.from_dict(DistanceEditDict, orient='index')
    df.index.name = 'FileName'
    df.to_excel('EasyOCR_Sorted_GPU.xlsx')

    total_end_time = time.time()
    total_process = total_start_time - total_end_time
    total_processing_time.append(total_process)
    print("TOTAL TIME: ", total_processing_time)


if __name__ == "__main__":
    # Create a CPUMonitor instance with a 0.5-second interval
    cpu_monitor = CPUMonitor(interval=0.5)
    cpu_monitor.start()  # Start monitoring CPU usage

    start_time = time.time()
    main()  # Run main
    end_time = time.time()

    # Stop monitoring CPU usage and calculate the average CPU and GPU usage
    avg_cpu_percent, avg_gpu_percent = cpu_monitor.stop()
    rss = get_process_info()  # Get the RAM usage

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Average CPU usage: {avg_cpu_percent:.2f}%")
    print(f"Average GPU usage: {avg_gpu_percent:.2f}%")
    print(f"RAM usage: {rss:.2f} MB")
