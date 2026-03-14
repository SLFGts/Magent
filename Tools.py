from datetime import datetime
import os
from pydoc import doc
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import requests
from langchain_core.tools import StructuredTool, create_retriever_tool
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

# 添加这两行，强制设置代理（本地代理端口为 7890）
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 自动加载 .env 文件
load_dotenv()

# ====================== 工具0：联网搜索工具 ======================
tavily_search_tool = TavilySearch(
    description="所有需要获取客观信息、时事新闻时，使用此工具进行搜索,但是需要明确说明使用了该工具。",
    max_results=5,
    include_answer="advanced",
)


# ====================== 工具1：城市名转经纬度 ======================
class GeocodingInput(BaseModel):
    location: str = Field(description="要查询经纬度的城市名，比如'上海'、'北京'")


def geocoding_func(location: str) -> dict[str, str]:
    """城市名转换经纬度，返回字典：{"lat": 纬度字符串, "lon": 经度字符串}"""
    print("开始转换地址为经纬度")
    base_url = "http://api.openweathermap.org/geo/1.0/direct?"
    api_key = os.getenv("OPENWEATHER_API_KEY")
    params = {
        "q": location,
        "appid": api_key,
        "limit": 1,
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {"lat": str(data[0]["lat"]), "lon": str(data[0]["lon"])}
    except Exception as e:
        return {"lat": "", "lon": f"经纬度查询失败：{str(e)}"}


geocoding_tool = StructuredTool.from_function(
    name="Geocoding",
    description="输入城市名，返回包含lat（纬度）、lon（经度）的字典，可用于获取WeatherQuery工具的参数",
    func=geocoding_func,
    args_schema=GeocodingInput,
)


# ====================== 工具2：根据经纬度查实时天气 ======================
class WeatherInput(BaseModel):
    # latitude: str = Field(description="纬度字符串，比如'31.2322758'")
    # longitude: str = Field(description="经度字符串，比如'121.4692071'")

    # 将两个参数合并为一个字符串参数，并告诉模型用逗号分隔
    lat_lon: str = Field(
        description="经纬度字符串，请用英文逗号分隔，比如'31.2322758,121.4692071'"
    )
    unit: str = Field(default="metric", description="温度单位，默认metric（摄氏度）")


def weather_query_func(lat_lon: str, unit: str = "metric") -> str:
    """经纬度查询天气，返回格式化的天气信息"""
    print("开始查询天气")

    # 增加内部解析逻辑
    try:
        latitude, longitude = [x.strip() for x in lat_lon.split(",")]
    except ValueError:
        return "参数错误：无法解析经纬度，请确保格式为 '纬度,经度'"

    if not latitude or not longitude:
        return "经纬度参数为空，无法查询天气"

    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    api_key = os.getenv("OPENWEATHER_API_KEY")
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": unit,
        "lang": "zh_cn",
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            return f"天气查询失败：{data.get('message', '未知错误')}"

        # 格式化结果
        return f"""
📍 {data['name']} 当前天气：
🌡️ 温度：{data['main']['temp']}°C（体感{data['main']['feels_like']}°C）
💧 湿度：{data['main']['humidity']}%
🌬️ 风速：{data['wind']['speed']} m/s
📝 状况：{data['weather'][0]['description']}
⏰ 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
    except Exception as e:
        return f"天气查询失败：{str(e)}"


weather_tool = StructuredTool.from_function(
    name="WeatherQuery",
    description="输入latitude（纬度）、longitude（经度）、unit（可选），返回天气信息",
    func=weather_query_func,
    args_schema=WeatherInput,
    # return_direct=True,
)

# ====================== 工具3：RAG工具 ======================
# 1. 自动获取当前 Tools.py 文件所在的文件夹绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 动态拼凑出 PDF 的路径（这样不管在 Mac 还是云端 Linux 都能精准找到！）
pdf_path = os.path.join(BASE_DIR, "绝密.pdf")

# 加载器
loader = PyPDFLoader(
    file_path=pdf_path,
)
# 加载
docs = loader.load()

# 拆分器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400,
)
# 拆分
chunks = splitter.split_documents(docs)


# 嵌入模型
embeder = OpenAIEmbeddings(
    model="text-embedding-3-large", base_url=os.getenv("OPENAI_BASE_URL")
)

# 自动获取当前 Tools.py 文件所在的文件夹绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 拼凑出 chroma_db 的绝对路径
persist_dir = os.path.join(BASE_DIR, "chroma_db")


# 创建向量存储
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeder,
    persist_directory=persist_dir,  # 持久化存储路径
)

# 从向量数据库中得到检索器
retriever = vector_store.as_retriever()
# 创建一个工具来检索文
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="ReportRetriever",
    description="需要查询检索绝密文件时，使用此工具检索，再返回信息",
)
