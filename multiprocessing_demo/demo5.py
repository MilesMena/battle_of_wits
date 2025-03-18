import concurrent.futures
import random
import time
import psutil
import csv
import os
import multiprocessing
import time


def worker(task_id):
    sleep_time = random.uniform(0.1, 5)
    time.sleep(sleep_time)
    message = f"Task {task_id} says {random.choice(['Hello', 'World', 'Foo', 'Bar'])}"
    worker_pid = os.getpid()
    return (task_id, worker_pid, message, round(sleep_time, 2))

def writer_callbacK(result):
    with lock:
        with open('output.csv', 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(result)
    print(f"Wrote result for Task {result[0]} by Worker {result[1]}")

def get_available_workers():
    cpu_usage = psutil.cpu_percent(interval=1)

    if cpu_usage < 50:
        return 4  # each one of these is a task in the command: top
    elif cpu_usage < 75:
        return 2
    else:
        return 1

def time_me(func):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()
        result=func(*args, **kwargs)
        end_time = time.time()
        print(end_time - start_time)
        return result
    return wrapper_function

@time_me
def main():
    num_tasks = 100

    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task ID", "Worker PID", "Message", "Sleep Time"])

    num_workers = get_available_workers()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker,range(num_tasks))

        for result in results:
            writer_callbacK(result)
    
    print("all tasks done. ")



    
if __name__ == "__main__":
    lock = multiprocessing.Lock()

    main()

    # 4 workers
    # 70.26170134544373