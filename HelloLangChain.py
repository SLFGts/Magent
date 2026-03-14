from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate


load_dotenv()  # 自动加载 .env 文件
# print(os.getenv("OPENAI_API_KEY"))
# print(os.getenv("OPENAI_BASE_URL"))
myLLM = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # 安全读取
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,
)

# 组成消息列表
messages = [
    SystemMessage(
        content="你是一个擅长人工智能相关学科的专家,但是你每说一句话结束后都要在后面加上一个哈哈哈"
    ),
    HumanMessage(content="请解释一下什么是机器学习？"),
]

# 流式输出
print("流式输出：")
for token in myLLM.stream(messages):
    print(
        token.content, end="", flush=True
    )  # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)
print()
print("流式输出结束")

# response = myLLM.invoke(messages)
# print(response.content)
# print(type(response))

PromptTemplate()


