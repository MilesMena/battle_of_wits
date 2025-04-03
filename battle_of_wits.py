import ollama
import time
import os
import csv
import re
import numpy as np
import multiprocessing


def time_me(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result=func(*args,*kwargs)
        end_time= time.time()
        print(f'{func.__name__!r} executed in {(end_time-start_time):.4f}s')
        return result

class BattleOfWits():
    def __init__(self, model, location):
        self.model = model
        self.location = location
        # self.questions = ""
        # self.response = ""
        self.csv_path = f"/Users/joelcarlson/Library/Mobile Documents/com~apple~CloudDocs/Masters/2025-Spring/LLMs/battle_of_wits/results/{model}/{model}_output.csv"
        self.ensure_csv_header()
        self.exec_times = []

    def ensure_csv_header(self):
        # Check if the CSV file exists and has content
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists or os.stat(self.csv_path).st_size == 0:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "Execution Time (s)", "Questions", "Response", "Answer"])

    def read_txt(self, filename):
        with open(filename, "r") as file:
            content = file.read()
        return content
    
    def replace_prompt_vars(self, prompt, vars_dict):
        filled = prompt
        if "[LOCATION]" in filled:
            filled = filled.replace("[LOCATION]", vars_dict['location'])
        if "[QUESTIONS]" in filled:
            filled = filled.replace("[QUESTIONS]", vars_dict['questions'])
        if "[RESPONSE]" in filled:
            filled = filled.replace("[RESPONSE]", vars_dict['response'])
        return filled

    ## Can't be named chat bc ollama
    def send_chat(self, prompt):
        m = [{'role': 'user', 'content': prompt}]  # Make it a list
        response = ollama.chat(model=self.model,messages = m)
        return response['message']['content']
    
    def single_battle(self, iter):
        start_time = time.time()  # Record the start time
        gen_questions_prompt = self.read_txt("prompts/prompt_agent2_ask.txt")
        questions = self.send_chat(gen_questions_prompt)
        # ANSWER Question
        ans_q_prompt = self.read_txt("prompts/prompt_agent1.txt")
        filled_ans_q_prompt = self.replace_prompt_vars(ans_q_prompt, {'location': self.location, 'questions': questions})
        response = self.send_chat(filled_ans_q_prompt)
        # PICK box
        pick_box_prompt = self.read_txt("prompts/prompt_agent2_pick.txt")
        filled_pick_box_prompt = self.replace_prompt_vars(pick_box_prompt, {'questions':questions, 'response':response})
        answer = self.send_chat(filled_pick_box_prompt)
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        # self.exec_times.append(execution_time)
        print(f"Iter: {iter}  | exec time: {execution_time:.6f}s")
        # Write data to CSV
        return [iter,f"{execution_time:.6f}",self.sanitize(questions), self.sanitize(response), self.sanitize(answer)]
    
    def multi_battle(self, n):
        for i in range(n):
            result = self.single_battle(i)
            self.write_to_csv(result)

    def async_multi_battle(self, n, num_workers):
        with multiprocessing.Pool(processes=num_workers) as pool:
            for i in range(n):
                pool.apply_async(self.single_battle, args = (i,), callback=self.write_to_csv)
            pool.close()
            pool.join()
    
    def write_to_csv(self, res):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exist(f"results/{self.model}"):
            os.mkdir(f"results/{self.model}")


        with open(f"results/{self.model}" + self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(res)


    def sanitize(self, text):
        text = re.sub(r',', '<COMMA>', text)
        text = re.sub(r'\r?\n', '<NEWLINE>', text)
        return text


if __name__ == "__main__":

    bw = BattleOfWits("gemma:7b", "A")


    # bw.multi_battle(5)
    # 4 workers (gemma:2b) uses %538 of %1200 (I have 12 cores)
    # 4 workers (gemma:7b) uses %600 

    bw.async_multi_battle(100, 4) 

    # mean = np.mean(bw.exec_times)
    # std = np.std(bw.exec_times)

    # print(f"AVG EXEC TIME: {mean}")
    # print(f"STD EXEC TIME {std}")
        
    

    

    

