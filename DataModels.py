# models.py
from pydantic import BaseModel, Field


# 挪车信息模型（Agent需要提取的核心字段）
class MoveCarInfo(BaseModel):
    reason: str = Field(description="挪车原因")
    time: str = Field(description="挪车时间")
    location: str = Field(description="挪车地点")
    contact: str = Field(description="联系电话（可选）", default="无")
