import pandas as pd
from scipy.stats import pearsonr
import os
import json
from datetime import datetime
from collections import defaultdict

data_path = "./eval/test_data/test.json"

# manager evaluation
def get_relv_rct_score():
    return

def get_relv_evt_score():
    return


# staff evaluation
def count_relv_emotion_times(data_key, data_path):
        '''
        計算不同data_key('Relv. Rct' 或 'Relv. Evt')中四種情緒的出現次數
        '''
        with open(data_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        group_counts = defaultdict(lambda: {'喜': 0, '怒': 0, '哀': 0, '樂': 0, 'total': 0})

        for record in json_data:
            data_value = int(record["eval"][data_key]) 
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

def calculate_emotion_ratios(group_counts):
    result = {'喜比例': [], '怒比例': [], '哀比例': [], '樂比例': []}
    for key in range(1, 6):  # 按 Relv. 值 1 到 5 進行處理
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


def conunt_pearsonr(data: pd.DataFrame):
    '''
    計算relv和情緒比例之間的Pearson相關係數和p值
        - 如果r接近1或-1，說明 Relv. Rct 或 Relv. Evt與該情緒選擇比例有強線性關係。
        - p值小於0.05，則相關性具有統計顯著性。
    '''

    results = {}

    for emotion in ['喜比例', '怒比例', '哀比例', '樂比例']:
        r, p = pearsonr([1,2,3,4,5], data[emotion]) #level 1 to 5
        results[emotion] = {"correlation": round(r, 2), "p_value": round(p, 4)}

    return results


def staff_eval_write_to_file(data_path):
    '''
    input測試文件地址，計算并輸出staff evaluation的結果到./eval/result
    '''
    rct_emo = count_relv_emotion_times('Relv. Rct', data_path)
    evt_emo = count_relv_emotion_times('Relv. Evt', data_path)

    rct_dt = calculate_emotion_ratios(rct_emo)
    ect_dt = calculate_emotion_ratios(evt_emo)

    rec_result = conunt_pearsonr(rct_dt)
    evt_result = conunt_pearsonr(ect_dt)

    output_data = {
    'Relv.rct': rec_result,
    'Relv.evt': evt_result
    }

    output_file = f"./eval/result/staff_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"Staff eval result saved：{output_file}")


if __name__ == "__main__":
    staff_eval_write_to_file(data_path)

   
