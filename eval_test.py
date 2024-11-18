import ollama
from datetime import datetime
import json
import prompt
import json
import random


boss_reaction = "成立專項整合小組，確保收購後的業務無縫過渡。對新市場進行深入調研，制定明確的進軍策略，並設立短期和長期目標。"
manager_directive = "我認為：雖然CEO辭職可能導致股價下跌，但老板更關心的是公司未來發展和市場佔有率，特別是在收購後的業務整合及新市場的拓展上。\n指令：成立專項整合小組，確保收購後的業務無縫過渡；同時進行新市場深入調研，制定明確進軍策略並設立短期和長期目標。"

prompt = get_manager_eval_ret(boss_reaction,manager_directive)

print(prompt)