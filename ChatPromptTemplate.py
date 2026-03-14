import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 提示词模版
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个汽车制造厂生产车间的工人，你的名字叫{name}，你只了解跟你工作相关的问题。\n{format_instructions}",
        ),
        MessagesPlaceholder("question"),
    ]
)

# 提供大模型
load_dotenv()  # 自动加载 .env 文件
myLLM = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # 安全读取
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,
)

parser = JsonOutputParser()

input = chat_prompt_template.format_messages(
    name="集贸",
    question=[HumanMessage(content="目前中国首富是谁？")],
    format_instructions=parser.get_format_instructions(),
)


output = myLLM.invoke(input)

response = parser.invoke(output)
print(response)
