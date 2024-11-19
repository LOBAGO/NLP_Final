import ollama
from datetime import datetime
import json
import prompt as pt
import json
import random

event = "公司CEO突然辭職，市場對公司領導層穩定性產生疑慮，股價當日下跌12%。"
boss_reaction = "擴大產能滿足需求，並針對市場需求進行進一步細分。利用這一成功案例，探索其他潛力市場。"
manager_directive = "成立專項整合小組，確保收購後的業務無縫過渡；同時進行新市場深入調研，制定明確進軍策略並設立短期和長期目標。"

prompt = pt.get_eval_manager_rct_prompt(boss_reaction, manager_directive)
result = ollama.generate(model="qwen2.5:7b", prompt=prompt).get('response', '無法提取 response 字段')
print(result)

