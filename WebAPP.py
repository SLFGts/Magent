import streamlit as st

# 直接从你的 Agent.py 中导入写好的 chat 函数
from Agent import chat

# 1. 设置网页的标题和图标
st.set_page_config(page_title="智能问答小助手", page_icon="❤️‍🔥", layout="centered")
st.title("❤️‍🔥 智能问答小助手 v2.0 (Web版)")
st.caption("集成了 实时天气、网络搜索 与 本地绝密知识库 的超级智能体")

# 2. 初始化 Streamlit 的会话状态 (Session State)
# Streamlit 每次刷新都会重头运行代码，所以我们要用 session_state 把历史聊天记录存起来显示在网页上
if "session_id" not in st.session_state:
    st.session_state.session_id = "web_user_001"  # 给网页用户发一个固定的身份证

if "messages" not in st.session_state:
    # 默认的第一条欢迎语
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是智能助手，有什么可以帮你的吗？"}
    ]

# 3. 渲染历史聊天记录（将之前存起来的消息一条条画在网页上）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. 处理用户输入并调用大模型
# st.chat_input 会在网页最下方生成一个极具现代感的聊天输入框
if user_input := st.chat_input("请输入您的问题... (比如：上海今天适合穿什么？)"):

    # a. 把用户的话画在网页上，并存入网页记忆
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # b. 调用大模型，并在网页上显示“思考中”的转圈加载动画
    with st.chat_message("assistant"):
        with st.spinner("🧠 正在思考并调用工具 (可能在检索绝密文件或查天气)..."):
            # 【核心逻辑】这里直接调用了你 Agent.py 里的函数！
            response = chat(user_input, session_id=st.session_state.session_id)

        # 思考完毕，把助手的回复画在网页上
        st.markdown(response)

    # c. 把助手的回复存入网页记忆
    st.session_state.messages.append({"role": "assistant", "content": response})
