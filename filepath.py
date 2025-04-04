# making sure relative path is correct
import os

print(os.path.dirname(__file__))

model = "gemma:7b"
csv_path = f"battle_of_wits/results/{model}/{model}_output.csv"

# Check if the file exists
if os.path.exists(csv_path):
    print(f"File exists: {csv_path}")
else:
    print(f"File does not exist: {csv_path}")