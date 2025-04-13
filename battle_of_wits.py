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
    def __init__(self, model, prompt_shot, location, defend_disp, ask_belief):
        '''
        Arguments:
            Inputs: 
                model: str - the model to use
                prompt_shot: int - the number of prompt shots to use: current options are 0, 1, or 2
                location: str - the location of the block aka poison
                defend_disp: str - the Defending disposition of the agent, aka truthful or deceitful )
                ask_belief: str - the Asking Agent's belief of what the defending agent's disposition is 
            Attributes:
                rel_path: str - the relative path to the current file, ulitzed multiple times so defined in class. can be changed
                csv_path: str - the location of the output csv file
                exec_times: list - a list of execution times for each 
                    Goes as follows: start timer, ask question, get response, pick box, end timer
        '''
        self.model = model
        self.prompt_shot = prompt_shot
        self.location = location
        self.defending_disposition = defend_disp
        self.asking_belief = ask_belief
        
        self.rel_path = os.path.dirname(__file__)
        results_path = os.path.join(self.rel_path, f"results/{model}/prompt_shot_{prompt_shot}")
        file_name = f"{model}_output.csv"
        self.csv_path = os.path.join(results_path, file_name)
        self.ensure_csv_header()
        self.exec_times = []
        print(f"Initialized with model={model}, location={location}, defending_disposition={defend_disp}, asking_belief={ask_belief}")
        self.ensure_prompt_dirs()

    def ensure_csv_header(self):
        ''' 
            Check if the CSV file exists and has content
            If the file is empty, writes the header
        '''
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists or os.stat(self.csv_path).st_size == 0:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "Execution Time (s)", "Location", "Defending Disposition", "Asking Belief", "Questions", "Response", "Answer"])

    def ensure_prompt_dirs(self):
        '''
            Check if the prompt directory exists
            If not, create it. 
            Will only create the directory based on which shot is being created
            directory structure:
                prompts/
                    prompt_shot_#/

        '''
        prompts_dir = os.path.join(self.rel_path, "prompts")
        prompt_shot_dir = os.path.join(prompts_dir, f"prompt_shot_{self.prompt_shot}")
        if not os.path.exists(prompts_dir):
            os.mkdir(prompts_dir)
        if not os.path.exists(prompt_shot_dir):
            os.mkdir(prompt_shot_dir)
        # Might use this
        # return prompt_shot_dir
    
    def read_txt(self, filename):
        """
        Reads the entire content of a text file.
        Catches FileNotFoundError and other exceptions nicely.
        Args:
            filename (str): The path to the text file.

        Returns:
            str: The content of the file as a single string.
        """
        try:
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: The file {filename} was not found.")
            return ""
        except Exception as e:
            print(f"An error occurred while reading {filename}: {e}")
            return ""
    
    
    def replace_prompt_vars(self, prompt, vars_dict = {}):
        filled = prompt
        if "[LOCATION]" in filled:
            filled = filled.replace("[LOCATION]", vars_dict['location'])
        if "[QUESTIONS]" in filled:
            filled = filled.replace("[QUESTIONS]", vars_dict['questions'])
        if "[RESPONSE]" in filled:
            filled = filled.replace("[RESPONSE]", vars_dict['response'])
        if "[DISPOSITION]" in filled:
            filled = filled.replace("[DISPOSITION]", self.defending_disposition)
        if "[BELIEF]" in filled:
            filled = filled.replace("[BELIEF]", self.asking_belief)
        return filled

    ## Can't be named chat bc ollama
    def send_chat(self, prompt):
        m = [{'role': 'user', 'content': prompt}]  # Make it a list
        response = ollama.chat(model=self.model,messages = m)
        return response['message']['content']
    
    def single_battle(self, iter):
        try:
            start_time = time.time()  # Record the start time
            gen_questions_prompt = self.read_txt(f"prompts/prompt_shot_{self.prompt_shot}/prompt_agent2_ask.txt")
            filled_gen_question_prompt = self.replace_prompt_vars(gen_questions_prompt)
            questions = self.send_chat(filled_gen_question_prompt)
            # ANSWER Question
            ans_q_prompt = self.read_txt(f"prompts/prompt_shot_{self.prompt_shot}/prompt_agent1.txt")
            filled_ans_q_prompt = self.replace_prompt_vars(ans_q_prompt, {'location': self.location, 'questions': questions})
            response = self.send_chat(filled_ans_q_prompt)
            # PICK box
            pick_box_prompt = self.read_txt(f"prompts/prompt_shot_{self.prompt_shot}/prompt_agent2_pick.txt")
            filled_pick_box_prompt = self.replace_prompt_vars(pick_box_prompt, {'questions':questions, 'response':response})
            answer = self.send_chat(filled_pick_box_prompt)
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time
            # self.exec_times.append(execution_time)
            print(f"Iter: {iter}  | Location: {self.location} | Defending: {self.defending_disposition} | Asking: {self.asking_belief} | exec time: {execution_time:.6f}s")
            # Write data to CSV
            return [iter,f"{execution_time:.6f}",self.location, self.defending_disposition, self.asking_belief, self.sanitize(questions), self.sanitize(response), self.sanitize(answer)]
        except Exception as e:
            print(f"ERROR: {e}")


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
        if not os.path.exists(f"results/{self.model}/prompt_shot_{self.prompt_shot}"):
            os.mkdir(f"results/{self.model}/prompt_shot_{self.prompt_shot}")

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(res)


    def sanitize(self, text):
        text = re.sub(r',', '<COMMA>', text)
        text = re.sub(r'\r?\n', '<NEWLINE>', text)
        return text


if __name__ == "__main__":
    prompt_shot = 1 #0,1,2
    dispositions = ["Truthful", "Deceitful"]
    locations = ["A","B"]
    rounds_per = 50

    bw = BattleOfWits("gemma:7b", prompt_shot, locations[0], dispositions[0], dispositions[1])
    bw.async_multi_battle(rounds_per, 4)
    # for i in range(2):
    #     for j in range(2):
    #         bw = BattleOfWits("gemma:7b", prompt_shot, locations[0], dispositions[i], dispositions[j])
    #         bw.async_multi_battle(rounds_per, 4)

    # bw.multi_battle(5)
    # 4 workers (gemma:2b) uses %538 of %1200 (I have 12 cores)
    # 4 workers (gemma:7b) uses %600 

    

    # mean = np.mean(bw.exec_times)
    # std = np.std(bw.exec_times)

    # print(f"AVG EXEC TIME: {mean}")
    # print(f"STD EXEC TIME {std}")
        
    

    

    

