# monitor_laptop.py
import psutil
import GPUtil
import time

def monitor_system():
    """Monitor laptop system resources"""
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    
    print(f"\n{'='*40}")
    print(f"RTX 3050 6GB Laptop Monitor")
    print(f"{'='*40}")
    print(f"GPU: {gpu.name}")
    print(f"  Load: {gpu.load*100:.1f}%")
    print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
    print(f"  Temp: {gpu.temperature}°C")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    print(f"\nCPU: {cpu_percent:.1f}%")
    if cpu_freq:
        print(f"  Freq: {cpu_freq.current:.0f} MHz")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.percent:.1f}%")
    print(f"  Used: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB")
    
    # Temperature (Linux)
    try:
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            for entry in temps['coretemp']:
                if 'Core' in entry.label:
                    print(f"  {entry.label}: {entry.current}°C")
    except:
        pass
    
    print(f"{'='*40}")