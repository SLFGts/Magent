import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

# 导入工具包
from Tools import weather_tool, geocoding_tool, tavily_search_tool, retriever_tool

# 准备工具列表
tools = [weather_tool, geocoding_tool, tavily_search_tool, retriever_tool]

# 自动加载 .env 文件
load_dotenv()

# 提供大模型
myLLM = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # 安全读取
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
)


# # 使用LangChain Hub中的官方ReAct提示模板
# prompt = hub.pull("hwchase17/react")
# # 创建ReAct代理
# agent = create_react_agent(llm=myLLM, tools=tools, prompt=prompt)

# # 使用适配 Tool Calling 的 Prompt
# prompt = hub.pull("hwchase17/openai-tools-agent")

# 自己定义 Prompt 模板（代替 hub.pull）
# 为什么要自己定义？因为我们需要在里面明确加上 chat_history（历史消息）的占位符
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个强大的AI助手，可以和用户对话。但是除了日常寒暄等对话内容外，所有其他问题都必须先使用检索工具检索答案，再使用搜索工具进行搜索后获取客观信息、时事新闻再回答。回答前必须声明使用了什么工具。",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # 这里专门用来放历史对话记忆
        ("user", "{input}"),  # 用户当前的输入
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),  # 这里放 Agent 调用工具的中间思考过程
    ]
)

# 使用 create_tool_calling_agent 替代 create_react_agent
agent = create_tool_calling_agent(llm=myLLM, tools=tools, prompt=prompt)

# 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示思考过程，调试时很有用
    max_iterations=3,  # 限制最大迭代次数，防止无限循环
    handle_parsing_errors=True,  # 自动处理解析错误
    early_stopping_method="generate",  # 提前停止策略
)

# 新增：设置记忆存储字典
# 这个字典用来保存不同用户的对话。在实际开发中，可以替换为 Redis 或数据库
store = {}


# 获取聊天历史的函数，历史记录库：BaseChatMessageHistory
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 获取对应的历史对话记录"""
    if session_id not in store:
        store[session_id] = (
            InMemoryChatMessageHistory()
        )  # 如果没有，就创建一个新的内存记忆
    return store[session_id]


# 新增：将 Agent 执行器与记忆模块“绑定”包装在一起
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,  # 你的"数据库"查询函数
    input_messages_key="input",  # 告诉它用户的输入对应 prompt 里的哪个变量
    history_messages_key="chat_history",  # 告诉它历史记忆要塞给 prompt 里的哪个变量
)


def chat(user_input: str, session_id: str = "default_user") -> str:
    """处理用户输入并返回助手回复（支持多轮对话记忆）"""
    try:
        # 调用包装了记忆的 agent，必须传入 config 告诉它当前是哪个 session_id
        response = agent_with_chat_history.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )
        return response["output"]
    except Exception as e:
        return f"抱歉，处理您的请求时出现错误：{str(e)}"


# 使用示例
def main():
    """主函数"""
    print("=" * 50)
    print("❤️‍🔥 智能问答小助手 v1.0 💝")
    print("=" * 50)
    print("输入'退出'或'quit'结束对话")
    print("-" * 50)

    # 我们给当前对话硬编码一个固定的 Session ID
    # 以后如果你做成 Web 接口，可以通过不同的 Session ID 来区分不同的用户！
    current_session_id = "user_Mary_Min"

    try:
        print("✅ 助手初始化完成，开始对话吧！\n")

        # 对话循环
        while True:
            try:
                user_input = input("你：").strip()

                if user_input.lower() in ["退出", "quit", "exit"]:
                    print("\n👋 再见！")
                    break

                if not user_input:
                    continue

                # 获取回复
                response = chat(user_input, session_id=current_session_id)
                print(f"\n助手：{response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出程序")
                break
            except Exception as e:
                print(f"❌ 处理输入时出错：{e}")

    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        print("请检查.env文件中的API密钥配置")
    except Exception as e:
        print(f"❌ 未知错误：{e}")


if __name__ == "__main__":
    main()
