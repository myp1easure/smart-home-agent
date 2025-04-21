import streamlit as st
import os
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
from openai import OpenAI
from rag import SmartHomeRAG
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# ------------------ 初始化 Vosk 模型 ------------------
vosk_model_path = "models/vosk-model-small-cn-0.22"
if not os.path.exists(vosk_model_path):
    st.warning("⚠️ 未找到 Vosk 中文模型，请将模型放在 models/vosk-model-small-cn-0.22 下")
    st.stop()

vosk_model = Model(vosk_model_path)

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="智能家居管家", layout="wide")
st.title("🏠 智能家居管家 (Smart Home Assistant)")

# ------------------ DashScope API 设置 ------------------
api_key = st.sidebar.text_input("🔑 DashScope API Key", type="password")
client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
# ------------------ 初始化 RAG 系统 ------------------
rag = SmartHomeRAG()

# ------------------ 页面标签页结构 ------------------
tab1, tab2 = st.tabs(["📋 设备状态 / 控制", "🎙️ 智能问答（语音支持）"])

# ------------------ 设备状态控制模块 ------------------

import json
import random
import os

# JSON file path for device state
STATE_FILE = "device_state.json"

# Initial device state (used if file doesn't exist)
default_device_state = {
    "空调": {"状态": "关闭", "温度": 27, "模式": "冷风", "风速": "中速"},
    "客厅灯": {"状态": "开启", "亮度": 80, "色温": "暖光"},
    "洗衣机": {"状态": "关闭", "模式": "标准", "剩余时间": 0},
    "扫地机器人": {"状态": "关闭", "电量": 85, "模式": "自动清扫"},
    "智能窗帘": {"状态": "关闭", "开合度": 0},
    "空气净化器": {"状态": "开启", "空气质量": "良好", "滤网寿命": 80}
}

def get_device_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"读取设备状态文件失败: {e}")
            return default_device_state
    else:
        # Create the file with default state if it doesn't exist
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_device_state, f, ensure_ascii=False, indent=2)
        return default_device_state

def save_device_state(state):
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存设备状态失败: {e}")
        return False

def control_device(device, action, **kwargs):
    device_state = get_device_state()
    if device in device_state:
        if action == "开启":
            device_state[device]["状态"] = "开启"
            # 根据设备类型设置特殊状态
            if device == "洗衣机" and "模式" in kwargs:
                device_state[device]["模式"] = kwargs["模式"]
                device_state[device]["剩余时间"] = random.randint(30, 90)
        elif action == "关闭":
            device_state[device]["状态"] = "关闭"
            if device == "洗衣机":
                device_state[device]["剩余时间"] = 0
        
        # 处理其他参数调整
        for key, value in kwargs.items():
            if key in device_state[device]:
                device_state[device][key] = value
        
        # Save changes to JSON file
        save_device_state(device_state)
        return True
    return False

with tab1:
    st.subheader("💡 设备状态与控制中心")
    states = get_device_state()
    
    # 添加刷新按钮
    if st.button("🔄 刷新设备状态", use_container_width=True):
        st.rerun()
    
    # 创建设备卡片网格布局 - 每行三个设备
    col1, col2, col3 = st.columns(3)
    
    for i, (device, info) in enumerate(states.items()):
        # 根据索引放置设备卡片在三列中
        with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3).container():
            # 使用彩色卡片风格，根据设备类型选择不同的图标
            device_icons = {"空调": "❄️", "客厅灯": "💡", "洗衣机": "🧺", 
                         "扫地机器人": "🤖", "智能窗帘": "🪟", "空气净化器": "🌬️"}
            icon = device_icons.get(device, "📟")
            
            card_color = "#e3f2fd" if info["状态"] == "开启" else "#f5f5f5"
            with st.container():
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: {card_color}; 
                     margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color: #1E88E5;">{icon} {device}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 设备状态显示，使用图标增强视觉效果
                device_status = info.get("状态", "未知")
                status_icon = "🟢" if device_status == "开启" else "🔴" if device_status == "关闭" else "⚪"
                st.markdown(f"**状态:** {status_icon} {device_status}")
                
                # 设备特定参数显示
                for key, value in info.items():
                    if key == "状态":
                        continue
                    
                    # 根据参数类型选择图标
                    param_icons = {
                        "温度": "🌡️", "模式": "⚙️", "风速": "💨", "亮度": "✨", 
                        "色温": "🎨", "剩余时间": "⏱️", "电量": "🔋", "开合度": "📏",
                        "空气质量": "🌈", "滤网寿命": "📅"
                    }
                    icon = param_icons.get(key, "ℹ️")
                    
                    # 格式化显示数值
                    if key in ["温度"]:
                        formatted_value = f"{value}°C"
                    elif key in ["亮度", "电量", "开合度", "滤网寿命"]:
                        formatted_value = f"{value}%"
                    elif key == "剩余时间" and value > 0:
                        formatted_value = f"{value}分钟"
                    else:
                        formatted_value = value
                        
                    st.markdown(f"**{key}:** {icon} {formatted_value}")
                
                # 设备特定控制界面
                if device == "空调":
                    st.markdown("##### 温度控制")
                    temp = st.slider("温度", min_value=16, max_value=30, value=info["温度"], key=f"temp_{device}")
                    if temp != info["温度"]:
                        control_device(device, info["状态"], 温度=temp)
                        st.rerun()
                    
                    st.markdown("##### 模式选择")
                    mode_col1, mode_col2, mode_col3 = st.columns(3)
                    mode = info["模式"]
                    if mode_col1.button("❄️ 冷风", key=f"cold_{device}"):
                        control_device(device, info["状态"], 模式="冷风")
                        st.rerun()
                    if mode_col2.button("🔥 暖风", key=f"hot_{device}"):
                        control_device(device, info["状态"], 模式="暖风")
                        st.rerun()
                    if mode_col3.button("🔄 自动", key=f"auto_{device}"):
                        control_device(device, info["状态"], 模式="自动")
                        st.rerun()
                
                elif device == "客厅灯":
                    st.markdown("##### 亮度调节")
                    brightness = st.slider("亮度", min_value=1, max_value=100, value=info["亮度"], key=f"bright_{device}")
                    if brightness != info["亮度"]:
                        control_device(device, info["状态"], 亮度=brightness)
                        st.rerun()
                
                elif device == "洗衣机" and info["状态"] == "开启":
                    st.markdown(f"##### 洗衣模式: {info['模式']}")
                    mode_col1, mode_col2 = st.columns(2)
                    if mode_col1.button("🧴 标准", key=f"standard_{device}"):
                        control_device(device, "开启", 模式="标准")
                        st.rerun()
                    if mode_col2.button("🧪 快洗", key=f"quick_{device}"):
                        control_device(device, "开启", 模式="快洗")
                        st.rerun()
                
                elif device == "智能窗帘":
                    if info["状态"] == "开启":
                        openness = st.slider("开合度", min_value=0, max_value=100, value=info["开合度"], key=f"open_{device}")
                        if openness != info["开合度"]:
                            control_device(device, "开启", 开合度=openness)
                            st.rerun()
                
                # 通用开关按钮
                button_col1, button_col2 = st.columns(2)
                
                if button_col1.button(f"✅ 开启", key=f"on_{device}", use_container_width=True):
                    control_device(device, "开启")
                    st.success(f"✨ {device}已成功开启！")
                    st.balloons()  # 添加气球动画效果
                    st.rerun()
                
                if button_col2.button(f"⛔ 关闭", key=f"off_{device}", use_container_width=True):
                    control_device(device, "关闭")
                    st.success(f"✨ {device}已成功关闭！")
                    st.rerun()

# ------------------ 智能问答（RAG + Vosk语音识别） ------------------
with tab2:
    st.subheader("输入方式")

    voice_text = ""
    q = queue.Queue()

    def record_callback(indata, frames, time, status):
        if status:
            st.warning(f"⚠️ 状态异常: {status}")
        q.put(bytes(indata))

    if st.button("🎤 使用Vosk进行语音识别"):
        st.info("🎧 开始录音，请在5秒内说话...")
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=record_callback):
            rec = KaldiRecognizer(vosk_model, 16000)
            while True:
                try:
                    data = q.get(timeout=5)
                except queue.Empty:
                    break
                if rec.AcceptWaveform(data):
                    break
            result = json.loads(rec.FinalResult())
            voice_text = result.get("text", "")
            if voice_text:
                st.success(f"🗣️ 识别结果：{voice_text}")
            else:
                st.error("⚠️ 没有识别到有效内容")

    user_query = st.text_input("💬 或直接输入指令：", value=voice_text)

    if user_query and client:
        with st.spinner("📚 正在查找相关资料并生成回复..."):
            docs = rag.retrieve(user_query)
            context = "\n".join(docs)

            prompt = (
                f"你是一个专业的中文智能家居助手，请结合以下说明书内容回答用户的问题：\n\n"
                f"{context}\n\n"
                f"用户提问：{user_query}"
            )

            try:
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "你是一个专业的中文智能家居助手。"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.success(answer)
            except Exception as e:
                st.error(f"❌ 生成回答失败: {str(e)}")
    elif not api_key:
        st.warning("⚠️ 请先在左侧输入 DashScope API Key")
