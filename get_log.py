import os
import json
import requests
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env
load_dotenv(override=True)

client = Client()

run_id = "8dfead97-996c-465f-929d-4e6bd4d4dc1e"

try:
    # 尝试通过 SDK 读取 (通常需要 Run 在你的项目下)
    run = client.read_run(run_id)
    run_data = run.dict()
    print("通过 SDK 成功读取 Run:", run.name)

except Exception as e:
    print(f"SDK 访问失败 (可能是公共 Run 或权限问题): {e}")
    print("尝试通过公开 API 访问...")
    
    # 尝试访问公开共享的 Run
    # 公开 Run 的 API 地址通常是: https://api.smith.langchain.com/public/<run_id>/run
    public_url = f"https://api.smith.langchain.com/public/{run_id}/run"
    response = requests.get(public_url)
    
    if response.status_code == 200:
        run_data = response.json()
        print("通过公开 API 成功读取 Run:", run_data.get("name"))
    else:
        print(f"公开 API 访问也失败了，状态码: {response.status_code}")
        print("响应内容:", response.text)
        run_data = None

def clean_run_data(raw_data):
    """
    从繁杂的 LangSmith Log 中提取核心对话逻辑
    """
    if not raw_data:
        return None

    # 1. 提取基础信息
    summary = {
        "run_name": raw_data.get("name"),
        "status": raw_data.get("status"),
        "error": raw_data.get("error"),
        "total_tokens": raw_data.get("total_tokens"), # 这个通常留着有个概念
        "latency_sec": None 
    }
    
    # 计算耗时
    if raw_data.get("start_time") and raw_data.get("end_time"):
        from datetime import datetime
        try:
            # 处理可能的带 Z 的 ISO 格式
            start_str = raw_data["start_time"].replace('Z', '+00:00')
            end_str = raw_data["end_time"].replace('Z', '+00:00')
            start = datetime.fromisoformat(start_str)
            end = datetime.fromisoformat(end_str)
            summary["latency_sec"] = (end - start).total_seconds()
        except Exception:
            pass

    # 2. 提取核心对话流 (最重要的部分)
    # LangGraph 通常把历史记录放在 outputs['messages'] 里
    messages = []
    
    # 获取输出中的消息列表
    raw_msgs = raw_data.get("outputs", {}).get("messages", [])
    
    # 如果输出里没拿到（比如出错提前结束），尝试从 inputs 里拿初始输入
    if not raw_msgs and "messages" in raw_data.get("inputs", {}):
        raw_msgs = raw_data["inputs"]["messages"]

    for msg in raw_msgs:
        # 有些消息可能是字典，有些可能是对象（取决于 SDK 版本/获取方式）
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "unknown")
        
        clean_msg = {
            "role": msg_type,  # human, ai, tool
            "content": content
        }
        
        # 处理工具调用
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if tool_calls:
            clean_msg["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} 
                if isinstance(tc, dict) else {"name": tc.name, "args": tc.args}
                for tc in tool_calls
            ]
        
        # 如果是工具的返回结果，提取工具名称
        name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)
        if name:
            clean_msg["tool_name"] = name
            
        messages.append(clean_msg)

    summary["conversation"] = messages
    return summary

if run_data:
    # --- 清洗数据 ---
    cleaned_log = clean_run_data(run_data)
    
    # 保存原始数据缓存
    with open("run_log_raw.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)

    # 保存清洗后的数据
    with open("run_log.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_log, f, ensure_ascii=False, indent=2)
    
    print("核心对话流已提取并保存到 run_log.json")
    print("原始完整日志已备份到 run_log_raw.json")
else:
    print("无法获取日志。")