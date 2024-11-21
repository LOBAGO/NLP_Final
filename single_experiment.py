import ollama
from datetime import datetime
import json
import prompt
import json
import random
from evaluation import evaluate, evaluate_with_pearson
from tqdm import tqdm
from main import get_random_data, manager_pipeline, staff_pipeline, clip_words

Boss_reaction_Path = 'data/Boss_reaction.json'
Event_Path = 'data/Event.json'
Staff_personality_Path = 'data/Staff_personality.json'
q7b = "qwen2.5:7b"
q3b = "qwen2.5:3b"
q14b = "qwen2.5:14b"


def single_experience(epochs):
    result_7b = []
    result_3b = []
    result_14b = []

    for i in tqdm(range(epochs)):
        event, = get_random_data(Event_Path, "content")
        boss_reaction, reaction_type = get_random_data(Boss_reaction_Path, "reaction", "type")
        manager_output = manager_pipeline(model_name=q3b,event=event,boss_order=boss_reaction)
        manager_directive = clip_words(marker="指令：",words = manager_output)
        result_3b.append({
            "id": i + 1,
            "input": {
                "event": event,
                'boss_reaction': boss_reaction,
                'type': reaction_type
            },
            "manager_output": manager_output,
            "manager_directive" : manager_directive,
        })
        manager_output = manager_pipeline(model_name=q7b,event=event,boss_order=boss_reaction)
        manager_directive = clip_words(marker="指令：",words = manager_output)
        result_7b.append({
            "id": i + 1,
            "input": {
                "event": event,
                'boss_reaction': boss_reaction,
                'type': reaction_type
            },
            "manager_output": manager_output,
            "manager_directive" : manager_directive,
        })
        manager_output = manager_pipeline(model_name=q14b,event=event,boss_order=boss_reaction)
        manager_directive = clip_words(marker="指令：",words = manager_output)
        result_14b.append({
            "id": i + 1,
            "input": {
                "event": event,
                'boss_reaction': boss_reaction,
                'type': reaction_type
            },
            "manager_output": manager_output,
            "manager_directive" : manager_directive,
        })
    output_file = f"./output/single_experiment/experiment_qwen2.5-3b.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_3b, f, ensure_ascii=False, indent=4)

    output_file = f"./output/single_experiment/experiment_qwen2.5-7b.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_7b, f, ensure_ascii=False, indent=4)

    output_file = f"./output/single_experiment/experiment_qwen2.5-14b.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_14b, f, ensure_ascii=False, indent=4)
    return

if __name__ == "__main__":
     single_experience(epochs=5)


    












