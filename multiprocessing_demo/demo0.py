import multiprocessing
import csv
import random
import time

def worker(i, lock):
    sleep_time = random.uniform(0.1, 1.0)  # Random delay
    time.sleep(sleep_time)
    random_message = f"Process {i} says {random.choice(['Hello', 'World', 'Foo', 'Bar'])}"
    
    # Write to CSV safely using lock
    with lock:
        with open('output.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, random_message, round(sleep_time, 2)])
    print(f"Process {i} finished in {round(sleep_time, 2)}s")

if __name__ == "__main__":
    # Create a lock for CSV writing
    lock = multiprocessing.Lock()

    # Clear or create the CSV with headers
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Process ID", "Message", "Sleep Time"])

    processes = []

    for i in range(10):  # Spawn 10 processes
        p = multiprocessing.Process(target=worker, args=(i, lock))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes complete. Check 'output.csv'")