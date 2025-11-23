#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟数据生成器 - 向仪表板发送模拟的视频流检测数据
"""

import asyncio
import json
import random
import time
from datetime import datetime
import websockets
import base64

# 生成模拟的检测数据
def generate_mock_detection(stream_id):
    """生成一个模拟的检测事件"""
    num_tracks = random.randint(0, 8)  # 随机生成0-8个跟踪目标

    tracks = []
    for i in range(num_tracks):
        # 生成随机边界框
        x1 = random.randint(0, 800)
        y1 = random.randint(0, 600)
        x2 = x1 + random.randint(50, 200)
        y2 = y1 + random.randint(50, 200)

        tracks.append({
            "track_id": i + 1,
            "class_id": random.choice(["person", "car", "truck", "bicycle", "motorcycle"]),
            "confidence": random.uniform(0.6, 0.99),
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    return {
        "stream": stream_id,
        "frame_id": random.randint(1000, 99999),
        "tracks": tracks,
        "fps": random.randint(20, 30),
        "received_at": datetime.now().isoformat(),
        "health": random.uniform(0.7, 1.0)
    }

async def send_mock_data():
    """连接到仪表板并发送模拟数据"""
    uri = "ws://localhost:8080/ws"

    # 定义32个视频流
    streams = [f"camera_{i:02d}" for i in range(1, 33)]

    try:
        async with websockets.connect(uri) as websocket:
            print("已连接到仪表板 WebSocket")

            # 先接收初始快照
            initial = await websocket.recv()
            print("收到初始快照")

            while True:
                # 每次更新多个流
                num_updates = random.randint(5, 15)
                selected_streams = random.sample(streams, num_updates)

                for stream_id in selected_streams:
                    # 生成检测事件
                    detection = generate_mock_detection(stream_id)

                    # 发送事件
                    message = {
                        "type": "event",
                        "payload": detection
                    }

                    await websocket.send(json.dumps(message))
                    print(f"发送数据: {stream_id} - {len(detection['tracks'])} 个目标")

                # 短暂延迟
                await asyncio.sleep(0.1)

                # 每隔一段时间发送完整快照
                if random.random() < 0.1:  # 10%概率发送快照
                    all_detections = [generate_mock_detection(s) for s in streams]
                    snapshot = {
                        "type": "snapshot",
                        "payload": {
                            "streams": all_detections
                        }
                    }
                    await websocket.send(json.dumps(snapshot))
                    print("发送完整快照")

                # 主循环延迟
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n停止发送模拟数据")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("开始发送模拟数据到仪表板...")
    print("按 Ctrl+C 停止")
    asyncio.run(send_mock_data())