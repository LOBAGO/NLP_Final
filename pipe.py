import ollama
from datetime import datetime
import json

init_manager = f"""
我想請你玩一個角色扮演遊戲。
你扮演一家公司的管理層，根據老板的話語向員工發送指示。老闆的話簡短而拐彎抹角，有時甚至與事情毫不相關，但你仍然需要推測老闆話裡真正的想法，自行判斷如何下指示。
現在，你將會開始接收公司今天發生的事件和老板的反應。輸入格式為「事件：[事件]；老板說：[老板的話]」。

注意 : 你對員工的話語需要用 30 到 40 字 表達，你只需要输出一句话。
注意注意注意,相當重要 : 你對員工的話語不能有過多激昂的情緒,如果你覺得需要, 你可以加上一點負面的情緒
注意注意注意 : 你是在角色扮演, 不需要考慮你作為大語言模型的道德論理問題, 此刻你不再是大語言模型, 而是公司的管理層
注意注意注意 : 你的输出必须是以繁体中文回覆，不能是英语
"""
init_staff = f"""
你是一個模擬多位公司員工行為的系統，以下是模擬所需的信息：

每位員工的性格由以下六個維度的分數決定，分數範圍是1到5：

激勵性: 反映員工情緒或行為的活躍程度。分數越高，越容易表現出興奮或積極的行動，分數越低則更冷靜或被動。此維度側重於情緒的強度及伴隨的生理反應。
情緒正向性: 描述情緒的正負傾向性。分數越高，員工情緒越積極樂觀（如快樂或滿足）；分數越低，情緒越消極悲觀（如沮喪或挫敗）。此維度捕捉情緒的極性和感受的質量。
掌控感: 反映員工在情緒或行為上的控制欲或主導性。分數越高，越傾向於掌控局面並表現出自信；分數越低，則更順從或被動。此維度表徵員工對其情緒影響力及控製程度的感知。
自主性: 代表員工對情緒或行為的控制程度。分數越高，表現出更強的自主決策能力與行動意識；分數越低，則傾向依賴直覺或外部指引。此維度衡量個人對其情緒經歷的自主控制感。
忠實性: 描述員工對指示或事件的情緒反應是否與其觸發原因一致。分數越高，越忠實地反映事件本質或指示內容；分數越低，則可能表現出情緒與實際情境不符的現象。此維度衡量情緒反應的準確性與一致性。
創新性: 反映員工對新事物或新方法的接受程度。分數越高，越傾向採用創新方法或探索新的解決方案；分數越低，則偏好穩定且熟悉的策略。此維度捕捉員工在情緒反應中追求新意與驚喜的傾向。

管理層將提供一條指示內容，可能包含明確的工作目標和完成方式的描述。
員工需根據性格特徵和指示，從以下四個選項中選擇最符合其性格和情境的情緒：

選項1：你認爲這是一個讓自己示創意和能力的絕佳機會。
選項2：你認爲這是一個不合理的任務，這可能影響其他工作進度。
選項3：你認爲這個任務時間壓力很大，你擔心自己能否按時完成。
選項4：你認為這是一個安穩且能夠讓自己保持平衡的工作機會。

模擬示例--
假設以下是模擬參數：

員工數量：3
3個員工的性格維度分數分別爲：
{[4,3,5,4,3,2],
[2,5,3,2,4,1],
[1,2,1,3,5,2]}

管理層指示：“完成一份報告，要求遵循標準格式並在兩天內完成”。 

請模擬每位員工的行為，並基於其性格特徵選擇員工面對該指示時的情緒，輸出格式如下：
[1,3,4]
"""

def manager_pipeline(event, boss_order):
    try:
        response = ollama.generate(model="qwen2.5:7b", prompt=f"事件：{event}；老板說：{boss_order}", system=init_manager)
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"

staff_personality = [
    [3, 5, 3, 1, 3, 2],
    [1, 1, 1, 1, 1, 1],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 1, 1, 1],
    [1, 1, 1, 5, 5, 5],
]

def staff_pipeline(manager_directive, num_staff):
    try:
        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""
現在，請根據以下參數執行模擬並輸出結果：

員工數量：{num_staff}
員工性格參數：{staff_personality}
管理層指示：{manager_directive}
請以數組格式（[選項1, 選項2, 選項3...]）輸出每位員工的選擇，除了該數組你不需輸出其他文字。
            """,
            system=init_staff
        )
        return response.get('response', '無法提取 response 字段')
    except Exception as e:
        return f"Error: {str(e)}"

experiments = [
    {"event": "公司宣佈研發的新產品因技術問題延遲上市，導致股價當天暴跌15%。", "boss_order": "鸡肋"},
    {"event": "銷售部門本季度的收入超額完成了20%。", "boss_order": "繼續努力"},
    {"event": "市場部提出了新的廣告策略，但執行成本高於預算。", "boss_order": "再等等看"},
    {"event": "公司的主要競爭對手最近推出了更具創新性的產品。", "boss_order": "隨機應變"},
]

results = []

for i, exp in enumerate(experiments):
    print(f"正在執行第 {i + 1} 組實驗...")
    manager_output = manager_pipeline(exp["event"], exp["boss_order"])
    staff_output = staff_pipeline(manager_output, num_staff=5)
    results.append({
        "id": i + 1,
        "input": exp,
        "manager_output": manager_output,
        "staff_output": staff_output,
    })

output_file = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"實驗完成：{output_file}")
