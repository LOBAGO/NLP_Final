import pandas as pd
from scipy.stats import pearsonr
import os
import json
from datetime import datetime
from collections import defaultdict
import prompt as pt
import ollama
from tqdm import tqdm
data_path = "./eval/relv/staff_eval_20241119_210310.json"
q7b = "qwen2.5:7b"
q3b = "qwen2.5:3b"
q14b = "qwen2.5:14b"

# evaluation
def get_relv_rct_score(model_name,boss_order,manager_directive):
    prompt = pt.get_eval_manager_rct_prompt(boss_order,manager_directive)
    relv_rct = ollama.generate(model=model_name,prompt=prompt).get('response', '無法提取 response 字段')
    return relv_rct

def get_relv_evt_score(model_name,event,boss_reaction):
    prompt = pt.get_eval_boss_evt_prompt(event,boss_reaction)
    relv_evt = ollama.generate(model=model_name,prompt=prompt).get('response', '無法提取 response 字段')
    return relv_evt

def get_relv_evt_socre_manager(model_name,event,manager_directive):
    prompt = pt.get_eval_boss_evt_prompt(event,manager_directive)
    relv_evm = ollama.generate(model=model_name,prompt=prompt).get('response', '無法提取 response 字段')
    return relv_evm

def get_eval_sentiment_score(model_name,event):
    prompt = pt.get_eval_sentiment_prompt(event)
    sentiment_score = ollama.generate(model=model_name,prompt=prompt).get('response', '無法提取 response 字段')
    return sentiment_score

# staff evaluation
def count_relv_emotion_times(data_key, data_path):
        '''
        計算不同data_key('Relv. Rct' 或 'Relv. Evt')中四種情緒的出現次數
        '''
        with open(data_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        group_counts = defaultdict(lambda: {'喜': 0, '怒': 0, '哀': 0, '樂': 0, 'total': 0})

        for record in json_data["results"]:
            data_value = int(record[data_key]) 
            staff_output = json.loads(record["staff_output"])  
            for emotion in staff_output:
                if emotion == 1:
                    group_counts[data_value]['喜'] += 1
                elif emotion == 2:
                    group_counts[data_value]['怒'] += 1
                elif emotion == 3:
                    group_counts[data_value]['哀'] += 1
                elif emotion == 4:
                    group_counts[data_value]['樂'] += 1

            group_counts[data_value]['total'] += len(staff_output)

        return group_counts

def calculate_emotion_ratios(group_counts, rating_len):
    result = {'喜比例': [], '怒比例': [], '哀比例': [], '樂比例': []}
    data_range = range(0, 2) if rating_len == 2 else range(1, rating_len+1) # 按 Relv. 值 1 到 5 進行處理
    for key in data_range:  
        total = group_counts[key]['total']
        if total > 0:
            joy_ratio = group_counts[key]['喜'] / total
            anger_ratio = group_counts[key]['怒'] / total
            sad_ratio = group_counts[key]['哀'] / total
            happy_ratio = group_counts[key]['樂'] / total
        else:
            joy_ratio = anger_ratio = sad_ratio = happy_ratio = 0.0
        result['喜比例'].append(joy_ratio)
        result['怒比例'].append(anger_ratio)
        result['哀比例'].append(sad_ratio)
        result['樂比例'].append(happy_ratio)
    return result


def compute_pearson_correlation_coefficient(data: pd.DataFrame, rating_len):
    '''
    計算relv和情緒比例之間的Pearson相關係數和p值
        - 如果r接近1或-1，說明 Relv. Rct 或 Relv. Evt與該情緒選擇比例有強線性關係。
        - p值小於0.05，則相關性具有統計顯著性。
    '''

    results = {}
    for emotion in ['喜比例', '怒比例', '哀比例', '樂比例']:
        r, p = pearsonr(list(range(1, rating_len+1)), data[emotion])
        results[emotion] = {"correlation": round(r, 2), "p_value": round(p, 4)}

    return results


def evaluate_with_pearson(data_path):
    '''
    input測試文件地址，計算并輸出staff evaluation的結果到./eval/pearson
    '''
    eval_list = [('eval_rct', 5), ('eval_evt', 5), ('eval_evm', 5)]
    with open(data_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    count = json_data["total_count"]
    output_data = {
        'input_file': data_path,
        'total_count': count
    }

    for eval_key, rating_len in eval_list:
        emotions = count_relv_emotion_times(eval_key, data_path)
        emotion_ratios = calculate_emotion_ratios(emotions, rating_len)
        result = compute_pearson_correlation_coefficient(emotion_ratios, rating_len)
        output_data[eval_key] = result

    output_file = f"./eval/pearson/staff_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"Staff eval result saved：{output_file}")


def evaluate(model_name,file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        ids = 1
        results = []
    for item in tqdm(data["results"], desc="evaluation", unit="item"):
        event = item['input']['event']
        boss_order = item['input']['boss_reaction']
        manager_directive = item['manager_directive']
        staff_output = item['staff_output']
        eval_rct = get_relv_rct_score(model_name=model_name,boss_order=boss_order,manager_directive=manager_directive)
        eval_evt = get_relv_evt_score(model_name=model_name,event=event,boss_reaction=boss_order)
        eval_evm = get_relv_evt_socre_manager(model_name=model_name,event=event,manager_directive=manager_directive)
        eval_sentiment =get_eval_sentiment_score(model_name=model_name,event=event)
        results.append({
            "id": ids,
            "staff_output": staff_output,
            "eval_rct": eval_rct,
            "eval_evt": eval_evt,
            "eval_evm": eval_evm,
            "eval_sentiment": eval_sentiment
        })
        ids += 1
    filename = file_path.split('/')[-1].split('.')[0]
    results_dict = {"total_count": ids - 1, "input_file": filename ,"model": model_name,"results": results}
    output_file = f"./eval/relv/relv_{filename}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    return output_file

if __name__ == "__main__":
    file = 'output/experiment_20241121_173335.json'
    eval_result = evaluate(file)
    evaluate_with_pearson('eval/relv/relv_experiment_20241121_173335.json')

   
