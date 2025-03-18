import multiprocessing
import csv
import random
import time

def worker(task_id):
    sleep_time = random.uniform(0.1, 1.0)
    time.sleep(sleep_time)
    message = f"Task {task_id} says {random.choice(['Hello', 'World', 'Foo', 'Bar'])}"
    print(f"Task {task_id} done in {round(sleep_time, 2)}s")
    return (task_id, message, round(sleep_time, 2))  # Ret


if __name__ == "__main__":
    num_workers = 4
    num_tasks = 20

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(worker, range(num_tasks))

    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task ID", "Message", "Sleep Time"])
        writer.writerows(results) 