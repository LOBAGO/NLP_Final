import ollama
from datetime import datetime
import json
import prompt
import json

model_name = "qwen2.5:7b"

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

staff_personality = [
    [3, 5, 3, 1, 3, 2],
    [1, 1, 1, 1, 1, 1],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 1, 1, 1],
    [1, 1, 1, 5, 5, 5],
]

def staff_pipeline(manager_directive, num_staff):
    try:
        start_marker = "指令："
        start_index = manager_directive.find(start_marker) + len(start_marker)
        manager_directive = manager_directive[start_index:].strip()
        print(manager_directive)
        response = ollama.generate(
            model=model_name,
            prompt=prompt.get_staff_prompt(num_staff, staff_personality, manager_directive),
            system=prompt.get_staff_system_prompt()
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

output_file = f"./output/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"實驗完成：{output_file}")