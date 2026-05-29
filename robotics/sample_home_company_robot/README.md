# sample_home_company_robot v2

## 功能概述

```
Sequence
├── ReceiveWhisperText          订阅 /whisper_out，非空写入 blackboard whisper_text
└── Fallback
    ├── Sequence (follow-me)
    │   ├── CheckFollowMe       检测 "follow me"（大小写不敏感）
    │   ├── SendFollowMeService 调用 /follow_me_start service，发送 0x01，等待 success=true
    │   └── StartFallDetection  发布 /start_fall_detection = true
    └── Sequence (LLM)
        ├── SendToLLM           发布 /llm_input，等待 /llm_output 回复，存入 blackboard llm_text
        └── SendToTTS           发布 /tts_input（llm_text）
```

## ROS 接口

| 方向 | 类型 | Topic / Service | 说明 |
|------|------|-----------------|------|
| Sub  | std_msgs/String | /whisper_out        | 语音识别输入 |
| Srv  | FollowMeStart   | /follow_me_start    | 发送 0x01，等待 bool success |
| Pub  | std_msgs/Bool   | /start_fall_detection | 跌倒检测开关 |
| Pub  | std_msgs/String | /llm_input          | 发送文本给 LLM |
| Sub  | std_msgs/String | /llm_output         | 接收 LLM 回复 |
| Pub  | std_msgs/String | /tts_input          | 发送文本给 TTS |

## 编译 & 运行

```bash
cp -r sample_home_company_robot_v2 ~/ros2_ws/src/sample_home_company_robot
cd ~/ros2_ws
colcon build --packages-select sample_home_company_robot
source install/setup.bash
ros2 launch sample_home_company_robot sample_home_company_robot.launch.py
```

## 测试

```bash
# 触发 follow me 流程
ros2 topic pub /whisper_out std_msgs/msg/String "{data: 'please follow me'}" --once

# 触发 LLM 流程
ros2 topic pub /whisper_out std_msgs/msg/String "{data: 'what is the weather today'}" --once

# 模拟 LLM 回复
ros2 topic pub /llm_output std_msgs/msg/String "{data: 'The weather is sunny.'}" --once
```
