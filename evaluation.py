import pandas as pd
from scipy.stats import pearsonr
import os
import json
from datetime import datetime

# manager evaluation
def get_relv_rct_score():
    ...

def get_relv_evt_score():
    ...



# staff evaluation


data = {
    'Relv. Rct': [4, 5, 3],
    'Relv. Evt': [3, 4, 5],
    '喜比例': [0.5, 0.4, 0.2],
    '怒比例': [0.1, 0.05, 0.3],
    '哀比例': [0.2, 0.1, 0.4],
    '樂比例': [0.2, 0.45, 0.1]
}
df = pd.DataFrame(data)

'''
- 如果r接近1或-1，說明 Relv. Rct 或 Relv. Evt與該情緒選擇比例有強線性關係。
- p值小於0.05，則相關性具有統計顯著性。
'''
def conunt_pearsonr(data: pd.DataFrame):
    results = {"Relv. Rct": {}, "Relv. Evt": {}}

    for metric in ['Relv. Rct', 'Relv. Evt']:
        for emotion in ['喜比例', '怒比例', '哀比例', '樂比例']:
            r, p = pearsonr(data[metric], data[emotion])
            results[metric][emotion] = {"correlation": round(r, 2), "p_value": round(p, 4)}
    
    output_file = f"./output/staff_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    for metric, emotions in results.items():
        print(f"{metric}:")
        for emotion, stats in emotions.items():
            print(f"  {emotion}: Correlation = {stats['correlation']}, p-value = {stats['p_value']}")

    print(f"\nResults saved")


if __name__ == "__main__":
    conunt_pearsonr(df)