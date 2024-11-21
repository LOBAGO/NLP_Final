import ollama
from datetime import datetime
import json
import prompt
import json
import random
from tqdm import tqdm

from evaluation import evaluate, evaluate_with_pearson
from tqdm import tqdm

Boss_reaction_Path = 'data/Boss_reaction.json'
Event_Path = 'data/Event.json'
Staff_personality_Path = 'data/Staff_personality.json'
q7b = "qwen2.5:7b"
q3b = "qwen2.5:3b"
q14b = "qwen2.5:14b"


def get_random_data(file_path, *datatype):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "events" in data and isinstance(data["events"], list) and len(data["events"]) > 0:
        # 從 events 列表中隨機選擇一個事件
        random_event = random.choice(data["events"])
        return tuple(random_event[dt] for dt in datatype)
    else:
        raise ValueError("data['events'] 不是有效的列表或沒有可選的事件")

def load_personality(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    staff_personality = [entry["personality"] for entry in data["personalities"]]
    num_staff = int(data['num_staff'])
    return num_staff, staff_personality

    
def manager_pipeline(model_name,event, boss_order):
    try:
        response = ollama.generate(
            model=model_name, 
            prompt=prompt.get_manager_prompt(event, boss_order), 
            system=prompt.get_manager_system_prompt()
        )
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"

num_staff, staff_personality = load_personality(Staff_personality_Path)

def staff_pipeline(model_name,manager_directive):
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt.get_staff_prompt(num_staff, staff_personality, manager_directive),
            system=prompt.get_staff_system_prompt()
        )
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"

def clip_words(marker,words):
        start_marker = marker
        start_index = words.find(start_marker) + len(start_marker)
        words = words[start_index:].strip()
        return words



def inference(model_name,epochs):
    results = []
    for i in tqdm(range(epochs), desc="inference", unit="epochs"): 
        event, = get_random_data(Event_Path, "content")
        boss_reaction, reaction_type = get_random_data(Boss_reaction_Path, "reaction", "type")
        manager_output = manager_pipeline(model_name=model_name,event=event, boss_order=boss_reaction)
        manager_directive = clip_words(marker="指令：",words = manager_output)
        staff_output = staff_pipeline(model_name=model_name,manager_directive=manager_directive)
        results.append({
            "id": i + 1,
            "input": {
                "event": event,
                'boss_reaction': boss_reaction,
                'type': reaction_type
            },
            "manager_output": manager_output,
            "manager_directive" : manager_directive,
            "staff_output": staff_output,
        })

    output_file = f"./output/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_dict = {"total_count": epochs, "model": model_name ,"results": results}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"實驗完成：{output_file}")
    return output_file

if __name__ == '__main__':
    output = inference(model_name=q7b,epochs=5)
    eval_output = evaluate(model_name=q7b,file_path=output)
    evaluate_with_pearson(eval_output)