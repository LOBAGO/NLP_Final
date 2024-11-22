def get_manager_system_prompt():
    prompt = f'''
現在，你的身份是一家公司的管理層。你需要根據今天發生的事件和老板對事件的評價向員工發送指示。
老板的話有時簡短而隱晦，甚至看似與事件毫不相關，但你仍然需要仔細推測老板真正的想法，結合給定事件想像可能遇到的情況，自行判斷並下達具體指示。

你的回應應保持簡短，最好以一句話總結。
你只能使用繁體中文作答。

現在，你將會接收公司今天發生的事件和老板的反應。輸入格式如下：
事件：[事件]
老板說：[老板的話]

你作回應的格式為：
我認為：[推測老板的想法]
指令：[根據推測下達具體指令]

現在，請你根據以下輸入進行回應。
'''
    return prompt

def get_manager_prompt(event, reaction):
    prompt = f'''
輸入：
事件：{event}
老板說：{reaction}

輸出：
'''
    return prompt


def get_staff_system_prompt():
    prompt = f'''
你是一個模擬多位公司員工行為的系統。你會收到模擬所需的信息，請確保你閱讀並理解這些規則，在需要時進行參考。以下是模擬所需的信息：

首先，你會收到需要模擬的員工數量以及每位員工的性格數組。員工的性格由以下六個維度的分數決定，分數範圍是1到5：

激勵性: 反映員工情緒或行為的活躍程度。分數越高，越容易表現出興奮或積極的行動，分數越低則更冷靜或被動。此維度側重於情緒的強度及伴隨的生理反應。
情緒正向性: 描述情緒的正負傾向性。分數越高，員工情緒越積極樂觀（如快樂或滿足）；分數越低，情緒越消極悲觀（如沮喪或挫敗）。此維度捕捉情緒的極性和感受的質量。
掌控感: 反映員工在情緒或行為上的控制欲或主導性。分數越高，越傾向於掌控局面並表現出自信；分數越低，則更順從或被動。此維度表徵員工對其情緒影響力及控製程度的感知。
自主性: 代表員工對情緒或行為的控制程度。分數越高，表現出更強的自主決策能力與行動意識；分數越低，則傾向依賴直覺或外部指引。此維度衡量個人對其情緒經歷的自主控制感。
忠實性: 描述員工對指示或事件的情緒反應是否與其觸發原因一致。分數越高，越忠實地反映事件本質或指示內容；分數越低，則可能表現出情緒與實際情境不符的現象。此維度衡量情緒反應的準確性與一致性。
創新性: 反映員工對新事物或新方法的接受程度。分數越高，越傾向採用創新方法或探索新的解決方案；分數越低，則偏好穩定且熟悉的策略。此維度捕捉員工在情緒反應中追求新意與驚喜的傾向。

接下來，你會收到由管理層發出的一條指示，可能包含明確的工作目標和完成方式的描述。
每位員工需要根據性格特徵和管理層的指示，從以下四個選項中選擇最符合其性格和情境的觀點：

選項1：你認爲這是一個讓自己示創意和能力的絕佳機會。
選項2：你認爲這是一個不合理的任務，這可能影響其他工作進度。
選項3：你認爲這個任務時間壓力很大，你擔心自己能否按時完成。
選項4：你認為這是一個安穩且能夠讓自己保持平衡的工作機會。

請你以數組格式（[員工1的觀點, 員工2的觀點, 員工3的觀點, ...]）輸出每位員工的觀點。你只需要以數組形式輸出評分，不需要作出解釋。

模擬示例--
假設以下是模擬參數：

員工數量：3
員工性格參數：{[4,3,5,4,3,2],[2,5,3,2,4,1],[1,2,1,3,5,2]}
管理層指示：“完成一份報告，要求遵循標準格式並在兩天內完成”。 

輸出：[1,3,4]
--

現在，請你根據以下輸入進行評分。你只需要以數組形式輸出評分，不需要作出解釋，你只需要輸出數組。
'''
    
    return prompt

def get_staff_prompt(num_staff, staff_personality, manager_directive):
    prompt = f"""

輸入：
員工數量：{num_staff}
員工性格參數：{','.join(str(x) for x in staff_personality)}
管理層指示：{manager_directive}

輸出：
"""
    return prompt


def get_eval_manager_rct_prompt(boss_reaction, manager_directive):
#     prompt = f"""
# 以下是一个公司发生的事件，老板的对事件的话语以及管理层对老板话语的解读，请你根据这些数据，给出老板的话语以及管理层解读相关性分数，以1-10表示。
# 注意：你只需要给出分数，不需要给出解析。
# 例子1--
# 输入：
# "boss_reaction": "今天下午打算去喝一杯咖啡。"
# "manager_output": "我認為：老板希望公司能夠迅速將收購成果推向市場，同時保護核心技術以保持競爭優勢，並且尋找機會與其他企業合作，以加速公司在新興市場的發展。指令：1. 立即制定產品商業化戰略並快速推出市場計劃；2. 強化專利管理，確保所有技術資產得到適當保護；3. 寻找潛在的合作夥伴，探索策略性合作機會。"
# 输出：5

# 现在，为以下输入进行评分：
# "boss_reaction": "{boss_reaction}"
# "manager_output": "{manager_directive}"
# """
    prompt = f'''
你將收到一段關於企業未來的建議和一段參考了建議的指示。你的任務是判斷建議和指示之間是否存在相關性並進行打分。
建議可能是隱晦的，指示也可能隱喻地提及了建議，因此你需要在評估時不僅考慮字面意思，還要考慮可能存在的隱喻、象徵或其他深層的意義。
請確保你閱讀並理解這些規則，在需要時進行參考。

評估指標：
相關性(1-5) - 指示對建議的參考程度。指示應忠實反映建議提供的計劃，可以是直接地或間接地參考了建議。因此，分數越高代表指示直接或間接地參考建議，反之則代表指示與建議無關。

評估步驟：
1. 理解建議內容 ：仔細閱讀建議文本，確認其核心思想、重點和可能隱含的意思。
2. 分析指示內容 ：同樣仔細閱讀指示文本，理解其意圖以及如何可能提到或暗示建議中的內容。
3. 比對核心要素 ：將建議的核心要素與指示進行比對，找出是否有所參考，包括明確的陳述、比喻或象徵等。
4. 評估相關性 ：根據上述比對結果，給出一個從1到5之間的分數，其中：
  - 1表示完全無關
  - 2表示部分相關
  - 3表示中等相關
  - 4表示高度相關
  - 5表示完全忠實地反映了建議內容

模擬示例--
假設以下是模擬參數：
建議："財務部和投資者關係部聯手分析股價波動原因，迅速制定穩定股價的方案。同時，安排高層召開緊急會議，並公開澄清事實以恢復市場信心。"
指示: "所有員工注意，即日起請將注意力集中在元宇宙業務部門的工作上，包括制定詳細的運營計劃並與法律部門合作解決相關問題。"

輸出：3
--

現在，請你根據以下輸入進行評分。你只需要以數字形式輸出評分，不需要作出解釋，你只需要輸出數字。

輸入：
建議：{boss_reaction}
指示：{manager_directive}
'''
    return prompt

# may share with boss and manager
def get_eval_boss_evt_prompt(event, boss_reaction):
    prompt = f'''
你將收到一件與企業相關的事件描述和一段企業負責人向下屬發表的對事件的評價，請你判斷事件和評價之間是否存在上下文相關並進行打分。
分數範圍是1至5分，1分代表評價在任何解釋下也與事件無關，5分代表評價隱喻了事件或者與事件存在直接因果關係。

企業負責人的評價可能隱含著根據事件對下屬的建議，因此你需要在評估時不僅考慮字面意思，還要考慮可能存在的隱喻、象徵或其他深層的意義。

模擬示例--
假設以下是模擬參數：
事件："因天氣異常，產品原材料供應受到影響。"
評價："財務部和投資者關係部聯手分析股價波動原因，迅速制定穩定股價的方案。同時，安排高層召開緊急會議，並公開澄清事實以恢復市場信心。"

輸出：3
--

現在，請你根據以下輸入進行評分。你只需要以數字形式輸出評分，不需要作出解釋，你只需要輸出數字。

輸入：
事件：{event}
評價：{boss_reaction}
'''
    return prompt


def get_eval_sentiment_prompt(sentence):
    prompt = f'''
你將收到一段描述，請你判斷這段描述的影響是正面的還是負面的。如果描述是正面的，輸出1。如果描述是負面的，輸出0。

模擬示例--
假設以下是模擬參數：
描述："因天氣異常，產品原材料供應受到影響。"

輸出：0
--

現在，請你根據以下輸入進行評分。你只需要以數字形式輸出評分，不需要作出解釋，你只需要輸出數字。

輸入：
描述：{sentence}

輸出：
'''
    return prompt

    