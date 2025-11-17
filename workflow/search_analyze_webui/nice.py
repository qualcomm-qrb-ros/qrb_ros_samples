from nicegui import ui
import threading

import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe('yaml_decide_node')
pubsub.subscribe('action_decide_node')
pubsub.subscribe('result')

# msg queue
import queue
q = queue.Queue()


# 初始化 TTS 引擎
import pyttsx3
final = ''
engine = pyttsx3.init()
def tts(text:str):
    engine.save_to_file(text, "./result.mp3" )
    engine.runAndWait()

# task area
with ui.card():
    text_area = ui.textarea(label='Command to Agent',value="Latest news of 陶喆", placeholder='enter your question').props('clearable')
    button = ui.button('Run Command', on_click=lambda:run_task())

with ui.column():
    with ui.row():
        with ui.card():
            a1 = ui.label(" Agent Task : ")
            a2 = ui.label(" NA ")

label_container = ui.row()

def run_task():
    task = text_area.value
    print(f"Running command: {task}")
    try:
        r.publish("input_task",task)
        a2.set_text(task)
        label_container.clear()
    except Exception as e:
        print(f"Exception occurred: {e}")


def listen_for_result():
    res = ''
    pubsub = r.pubsub()
    pubsub.subscribe('result')
    for message in pubsub.listen():
        if message['type'] == 'message':
            res = message['data']
            
            # handle tts
            final = res
            threading.Thread( target= tts(final) ).start()

            q.put(res)

def listen_for_tool():
    res = ''
    pubsub = r.pubsub()
    pubsub.subscribe('action_decide_node')
    for message in pubsub.listen():
        if message['type'] == 'message':
            res = message['data']
            q.put(res)

def listen_for_exec_tool_result():
    res = ''
    pubsub = r.pubsub()
    pubsub.subscribe('exec_tool_result')
    for message in pubsub.listen():
        if message['type'] == 'message':
            res = message['data']
            q.put(res)

def listen_for_response():
    res = ''
    pubsub = r.pubsub()
    pubsub.subscribe('response')
    for message in pubsub.listen():
        if message['type'] == 'message':
            res = message['data']
            q.put(res)


def listen_for_decision():
    res = ''
    pubsub = r.pubsub()
    pubsub.subscribe('yaml_decide_node')
    for message in pubsub.listen():
        if message['type'] == 'message':
            res = message['data']
            q.put(res)

def dis_msg():
    while True:
        ui.label(q.get())

threading.Thread(target=listen_for_result, daemon=True).start()
threading.Thread(target=listen_for_tool, daemon=True).start()
threading.Thread(target=listen_for_exec_tool_result, daemon=True).start()
threading.Thread(target=listen_for_response, daemon=True).start()
threading.Thread(target=listen_for_decision, daemon=True).start()


def check_and_create_label():
    if not q.empty():
        item = q.get_nowait()
        with label_container:
            with ui.card():
                ui.label(f' [\U0001F600] Agent回复: {item}')
    else:
        return


with ui.card():
    ui.button('朗读', on_click=lambda: ui.audio("./result.mp3"))


ui.timer(0.5, check_and_create_label)

ui.run()