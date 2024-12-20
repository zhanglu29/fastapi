from typing import List, Dict
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from langchain_groq import ChatGroq

# Initialize FastAPI app
app = FastAPI(title="Intent Recognition API",
              description="API for recognizing intents from user input")


# 定义输出模式
class Slot(BaseModel):
    """槽位信息"""
    name: str = Field(description="槽位名称")
    value: str = Field(description="槽位值")


class Intent(BaseModel):
    """意图及其槽位"""
    name: str = Field(description="意图名称")
    slots: Dict[str, str] = Field(description="意图的槽位信息")


class IntentResponse(BaseModel):
    """包含多个意图的响应"""
    intents: List[Intent] = Field(description="识别出的意图列表")


# 定义请求模型
class IntentRequest(BaseModel):
    text: str = Field(description="需要识别意图的输入文本")


def create_intent_prompt():
    # 定义系统提示和示例
    system_template = """你是一个意图识别助手。分析用户输入并识别其中包含的意图和相关信息。
    将结果格式化为JSON，包含意图名称和相关槽位信息,注意输出不要多余内容只要示例中 json 部分。

    示例输入：
    "我要去深圳会展中心，请将该位置定位发给微信好友张三，并帮我打个车到那里"

    示例输出：
    {{
        "intents": [
            {{
                "name": "send_location",
                "slots": {{
                    "location": "深圳会展中心",
                    "platform": "微信",
                    "recipient": "张三"
                }}
            }},
            {{
                "name": "call_taxi",
                "slots": {{
                    "destination": "深圳会展中心"
                }}
            }}
        ]
    }}"""

    human_template = "{text}"

    # 创建提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    return prompt_template.partial()


# 初始化环境变量和模型
def init_model():
    # 从环境变量获取 API keys
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    print("langchain_api_key:", langchain_api_key)
    print("groq_api_key", groq_api_key)

    # 检查必要的 API keys 是否存在
    if not langchain_api_key:
        raise ValueError("LANGCHAIN_API_KEY environment variable is not set")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    # 设置环境变量
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key

    return ChatGroq(model="llama-3.2-90b-vision-preview")


# 创建全局变量 test
model = init_model()
prompt = create_intent_prompt()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Intent Recognition API is running",
        "version": "1.0.0"
    }


@app.post("/recognize_intent", response_model=IntentResponse)
async def recognize_intent(request: IntentRequest):
    try:
        # 创建提示
        prompt_value = prompt.invoke({"text": request.text})
        # 获取模型响应
        response = model.invoke(prompt_value)

        # 将字符串响应解析为 JSON
        import json
        result = json.loads(response.content)

        # 使用 Pydantic 模型验证和转换响应
        intent_response = IntentResponse(**result)
        return intent_response
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail="Invalid JSON response from model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)