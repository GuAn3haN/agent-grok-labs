from threading import Thread
from modelscope import (AutoModelForCausalLM, AutoTokenizer)
from transformers import TextIteratorStreamer
device = "cuda"  # 将模型加载到哪个硬件，此处为GPU
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat", # 模型文件夹路径
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
while True:
    user_input = input("请输入问题（q退出）：")
    if user_input.lower() == "q":
        print("exit")
        break
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        count = 0
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)
        print()
    except Exception as e:
        print(f"出错了：{str(e)}")