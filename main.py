import ollama
from datetime import datetime
import json
import prompt
import json
import random

from evaluation import evaluate, evaluate_with_pearson

model_name = "qwen2.5:7b"
Boss_reaction_Path = 'data/Boss_reaction.json'
Event_Path = 'data/Event.json'
Staff_personality_Path = 'data/Staff_personality.json' 

def get_random_data(file_path,datatype):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "events" in data and isinstance(data["events"], list) and len(data["events"]) > 0:
        # 從 events 列表中隨機選擇一個事件
        random_event = random.choice(data["events"])
        return random_event[datatype]
    else:
        raise ValueError("data['events'] 不是有效的列表或沒有可選的事件")

def load_personality(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    staff_personality = [entry["personality"] for entry in data["personalities"]]
    num_staff = int(data['num_staff'])
    return num_staff, staff_personality

    
def manager_pipeline(event, boss_order):
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

def staff_pipeline(manager_directive):
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt.get_staff_prompt(num_staff, staff_personality, manager_directive),
            system=prompt.get_staff_system_prompt()
        )
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"

def boss_manager_eval(boss_reaction, manager_directive):
    try:
        response = ollama.generate(
            model= model_name,
            prompt= prompt.get_eval_manager_rct_prompt(boss_reaction, manager_directive)
        )
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"


def inference():
    results = []
    epochs = 1
    for i in range(epochs):
        print(f"正在執行第 {i + 1} 組實驗...")
        event = get_random_data(Event_Path, datatype="content")
        boss_reaction = get_random_data(Boss_reaction_Path, datatype="reaction")
        manager_output = manager_pipeline(event, boss_reaction)
        start_marker = "指令："
        start_index = manager_output.find(start_marker) + len(start_marker)
        manager_directive = manager_output[start_index:].strip()
        staff_output = staff_pipeline(manager_directive)
        results.append({
            "id": i + 1,
            "input": {
                "event": event,
                'boss_reaction': boss_reaction
            },
            "manager_output": manager_output,
            "manager_directive" : manager_directive,
            "staff_output": staff_output,
        })

    output_file = f"./output/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"實驗完成：{output_file}")
    return output_file

if __name__ == '__main__':
    output = inference()
    eval_output = evaluate(output)
    evaluate_with_pearson(eval_output)