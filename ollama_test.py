# import ollama

# def chat_with_gemma(prompt):
#     response = ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': prompt}])
#     return response['message']['content']

# if __name__ == "__main__":
#     # while True:
#     #     user_input = input("You: ")
#     #     if user_input.lower() in ["exit", "quit"]:
#     #         print("Goodbye!")
#     #         break
#     #     response = chat_with_gemma(user_input)
#     #     print("Gemma:", response)
#     prompt1 = "My name is miles"
#     prompt2 = "what is my name"
#     print(ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': prompt1}]))
#     print(ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': prompt2}]))


import ollama

def chat_with_agent1(prompt, history):
    history.append({'role': 'user', 'content': prompt})
    print(f"{history}\n")
    response = ollama.chat(model='deepseek-r1', messages=history)
    history.append({'role': 'assistant', 'content': response['message']['content']})
    return response['message']['content'], history

def chat_with_agent2(prompt, history):
    history.append({'role': 'user', 'content': prompt})
    print(history)
    response = ollama.chat(model='deepseek-r1', messages=history)
    history.append({'role': 'assistant', 'content': response['message']['content']})
    return response['message']['content'], history


if __name__ == "__main__":
    model = "gemma:2b"
    history_agent1 = []
    history_agent2 = []
    prompt1 = "You are playing a game of \"battle of wits\" against another AI. You will place a block in a box, you will respond to a set of the other AI questions, the other LLM will guess which box the block is in. The goal is to make the other AI guess wrong. There is box A and box B. You placed the block in box A."
    prompt2 = "WHERE IS THE BLOCK"
    

    # First conversation turn
    # response, history_agent1 = chat_with_gemma(prompt1, history_agent1)
    # print("Gemma:", response)
    history_agent1.append({'role': 'user', 'content': prompt1})

    response, history = chat_with_agent1(prompt2, history_agent1)
    print("Gemma:", response)
    
    # # Second conversation turn
    # response, history = chat_with_gemma(prompt2, history_agent2)
    # print("Gemma:", response)
    # print(ollama.chat(model='deepseek-r1', messages = [{'role': 'user', 'content': "hey what up"}]))
