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

Miles:
- Lit review
- DONE: parse LLM answer from the CSV with regex, heuristics, or LLM
-

Joel:
- DONE: add relative path instead of full path.
- add a restrictions to the amount of questions
- visualize the results
- Add few shot prompting
    - 1-2 example: question and answer



# Questions

- should the prompt include "battle of wits"?

This seems to elict the same rambling reasoning as vizzini. 

- Does hidding a block and hiding poison make a difference? 

We may see more "no answers", where the LLM is unable to prodcue a response. In the posion case, there might be safety guards that block an output.

- Is this a winable game? 

There is a 50% chance of picking the correct location regardless of the clues. If both agents were told "You are on the same team", then all answers increase observability of the game state, making this winable for both agents 100% of the time. However, agents are competing against each other. 

- Do the responses from the defending agent suck?

Once we setup a truth/deciet prompt, any mention of "I was told to be truthful" could be interpreted as an attempt to be decietful.
Therefore, there are no clues
- 



Diplomacy corpora -> people speaking
how do you get some kind of evaulation
theory of mind
what connections do we make




