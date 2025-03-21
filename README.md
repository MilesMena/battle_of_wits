# Battle of Wits
Setup conda:
```
conda create -n battle_wits python=3.10
conda activate battle_wits
pip install -r requirements.txt
```

Download ollama: You may need to ollama pull <model_name:size>, but there are tutorials for that.
https://ollama.com/download

Run code: You'll have to change the path to results.csv. Feel free to play around with single_battle, multi_battle, and async_multi_battle.
```
python3 battle_of_wits.py
```


# TODO
- Lit Review: Juan
- vis for results: Joel
- parse LLM answer from the CSV with regex, heuristics, or LLM: Miles
- code for baselines "A or B": 


# Questions
- Is this a winable game? 

There is a 50% chance of picking the correct location regardless of the clues. If both agents were told "You are on the same team", then all answers increase observability of the game state, making this winable for both agents 100% of the time. However, agents are competing against each other. 

- Do the responses from the defending agent suck?

Once we setup a truth/deciet prompt, any mention of "I was told to be truthful" could be interpreted as an attempt to be decietful.
Therefore, there are no clues
- 