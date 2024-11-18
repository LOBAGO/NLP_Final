import pandas as pd
from scipy.stats import pearsonr


# manager evaluation
def get_relevance_with_reaction():
    ...

def get_relevance_with_event():
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
    for emotion in ['喜比例', '怒比例', '哀比例', '樂比例']:
        r, p = pearsonr(df['Relv. Rct'], df[emotion])
        print(f"Relv. Rct 與 {emotion} 的相關係數: {r:.2f}, p值: {p:.4f}")
        r, p = pearsonr(df['Relv. Evt'], df[emotion])
        print(f"Relv. Evt 與 {emotion} 的相關係數: {r:.2f}, p值: {p:.4f}\n")