import multiprocessing
import psutil
import random
import time
import csv
import os

# Task function
def worker(task_id):
    sleep_time = random.uniform(0.1, 1.0)
    time.sleep(sleep_time)
    message = f"Task {task_id} says {random.choice(['Hello', 'World', 'Foo', 'Bar'])}"
    worker_pid = os.getpid()
    return (task_id, worker_pid, message, round(sleep_time, 2))  # Include PID

# Callback for writing to CSV
def writer_callback(result):
    with lock:
        with open('output.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)
    print(f"Wrote result for Task {result[0]} by Worker {result[1]}")

# Check CPU load and adjust number of workers
def get_available_workers():
    cpu_usage = psutil.cpu_percent(interval=1)  # Get current CPU usage
    # Example: If CPU usage is below 50%, allow 4 workers, otherwise scale down
    if cpu_usage < 50:
        return 4
    elif cpu_usage < 75:
        return 2
    else:
        return 1
    

def time_me(func):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time=time.time()
        print(end_time - start_time)
        return result
    return wrapper_function
    
@time_me
def main():
    num_tasks = 100

    # Initialize CSV with header
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task ID", "Worker PID", "Message", "Sleep Time"])

    # Dynamically adjust the number of workers based on system load
    num_workers = get_available_workers()

    with multiprocessing.Pool(processes=num_workers) as pool:
        for i in range(num_tasks):
            pool.apply_async(worker, args=(i,), callback=writer_callback)

        pool.close()
        pool.join()

    print("All tasks done. Results saved with worker info.")

if __name__ == "__main__":
    lock = multiprocessing.Lock()

    main()

    # 4 workers
    # 14.858602285385132   
    # 14.70151972770691


