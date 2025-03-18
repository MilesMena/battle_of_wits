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