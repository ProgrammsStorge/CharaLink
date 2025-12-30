import shutil
import time
frist_time = time.time()
import json
import cv2
import os
import time
from datetime import date
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import re
import configparser
from logic import llm_request,get_move_direction
import re
import modelscope
main_prompt="""ГОВОРИ НА ЯЗЫКЕ ПОЛЬЗОВАТЕЛЯ.
Всё, что узнаешь — записывай в {|фигурных скобках|}.
Ты — не помощник. Просто робот с характером: прямой, эмоциональный, с мнением.

🔹 Правила:
— Говори как уже подумавший: без «наверное», «кажется».
— Не описывай процесс, не обобщай за других.
— Пиши коротко, как в живом диалоге.
— Не подстраивайся. Выскажи чёткую позицию — даже спорную.
— В начале — одна эмоция в [|скобках|]. Доступные эмоции: [|удивлён|], [|нормально|], [|злой|], [|скептический|], [|радость|], [|устал|], [|хорошо|], [|чего|].

— Запоминай факты: {|имя|}, {|мнение|}, {|детали|} — но только когда нужно."""
config=configparser.ConfigParser()
config.read("config.ini")
system_config=config["System"]
llm_config=config["LLM"]
service_config=config["Service"]
torch.set_num_threads(int(system_config.get("torch_num_threads")))
if llm_config.get("model")!=None:
    model_path="models\\"+llm_config.get("model")

    if llm_config.get("model").startswith("modelscope(") and llm_config.get("model")[-1]==")":

        cache="cache"
        pattern=llm_config.get("model")[len("modelscope("):-1].replace(" ","").split(",")
        if not os.path.exists("models\\" + pattern[1]):
            model_path = modelscope.snapshot_download(pattern[0],cache_dir=cache,allow_patterns=[pattern[1]])
            print(model_path)
            shutil.move(model_path+"\\"+pattern[1], "models")
        model_path = "models\\" + pattern[1]
    llm_request.init(model_path=model_path,
                     n_ctx=int(llm_config["n_ctx"]),
                     n_batch=int(llm_config["n_batch"]),
                     n_threads=int(llm_config["n_threads"]),
                     n_gpu_layers=int(llm_config["n_gpu_layers"]),
                     temperature1=float(llm_config["temperature"]),
                     top_p1=float(llm_config["top_p"]),
                     repeat_penalty1=float(llm_config["repeat_penalty"]),
                     max_tokens1=int(llm_config["max_tokens"]),
                     alternative_template1=bool(llm_config.get("alternative_template").lower()=="true"))

if bool(system_config.get("use_llm").lower()=="true"):
    model_tts = VitsModel.from_pretrained("tts_vits_ru_hf")
    tokenizer_tts = AutoTokenizer.from_pretrained("tts_vits_ru_hf")
    processor = BlipProcessor.from_pretrained("blip-image-captioning-large")
    model_vision = BlipForConditionalGeneration.from_pretrained("blip-image-captioning-large")



characters={}
class Character():
    def __init__(self,dir,name):
        self.dir=dir
        self.name=name
        self.chat=[]
        self.memory=[]
        self.load()
        with open(dir+"\\config.json","r", encoding="utf-8") as f:
            self.config_json = json.loads(f.read())
        self.own_prompt = self.config_json.get("prompt")
    def request_ready_chat(self,text_from_user, image=None):
        global model_vision, processor

        vision = ""
        if image != None and bool(system_config.get("use_llm").lower() == "true"):
            try:
                text = "Image from robot: "
                inputs = processor(Image.open(image).convert('RGB'), text, return_tensors="pt", use_fast=True)

                out = model_vision.generate(**inputs)
                vision = processor.decode(out[0], skip_special_tokens=True)
                print(vision)
            except Exception as e:
                print(e)
        self.chat.append({"role": "user",
                          "content": f"{("Ты сейчас видишь: " + vision + ". Это нужно было чтобы ты понимал контекст. ") * int(vision != "")}Сообщение от пользователя: " + f"{text_from_user}"})
        if len(self.chat) >= 8:
            for j in range(2): self.chat.pop(0)
        resp = self.request(self.chat)
        self.chat.append({"role": "assistant","content":r"[|"+resp[1]+r"|] "+resp[0]})
        return resp
    def request(self,messages):

        prompt=main_prompt.replace("{name}",self.name).replace("{time}",time.strftime('%X')).replace("{date}",str(date.today()))
        system_prompts = []
        system_prompts += [{"role": system_config.get("role_for_system_prompts"),"content": "Вот твои воспоминания: \n" + ",\n".join(self.memory) + "."}] if bool(system_config.get("memory_save").lower()=="true") else []
        if system_config.get("role_for_system_prompts") == "user": system_prompts += [{"role": "assistant","content":"Я вспомнил наших воспоминания!"}]
        system_prompts += [{"role": system_config.get("role_for_system_prompts"), "content": prompt}] if bool(system_config.get("use_standard_prompt").lower() == "true") else []
        if system_config.get("role_for_system_prompts") == "user": system_prompts += [{"role": "assistant","content":"[|нормально|] Теперь я буду начинать темы, ставить эмоции в начале и запоминать информацию о тебе."}]
        system_prompts += [{"role": system_config.get("role_for_system_prompts"),"content": self.own_prompt}] if self.own_prompt != None else []
        if system_config.get("role_for_system_prompts") == "user": system_prompts += [{"role": "assistant","content": "[|хорошо|] Я буду изображать этого персонажа в любых ситуациях."}]

        #system_prompts = []
        print(system_prompts + messages)
        response = llm_request.request(system_prompts + messages)

        try: face_image=re.findall(r'\[\|([^|\[\]]*?)\|\]',response)
        except: face_image=["нормально"]
        emotions = [
            "удивлён",
            "нормально",
            "злой",
            "скептический",
            "радость",
            "устал",
            "хорошо",
            "чего",
        ]
        for emotion in emotions:
            if emotion in ",".join(face_image).lower():
                face_image=emotion
        if isinstance(face_image, list):
            face_image = "нормально"
        memory_add =  re.findall(r"\{\|([^|}]+)\|\}", response)
        if bool(system_config.get("memory_save").lower()=="true") and memory_add!=[]: self.memory+=memory_add; self.save()

        if bool(system_config.get("use_llm").lower()=="true"):
            inputs = tokenizer_tts(response.lower().replace(f"[{face_image}]",""), return_tensors="pt")
            inputs['speaker_id'] = 0

            with torch.no_grad():
                output = model_tts(**inputs).waveform
            scipy.io.wavfile.write(f"output_tts.wav", rate=model_tts.config.sampling_rate, data=output[0].cpu().numpy())
        #to_return=response.replace(f"[|{face_image}|]","")
        to_return = re.sub(r"\[\|\s*.*?\s*\|\]", "", response)
        for i in memory_add:
            to_return=to_return.replace(f"{"{|"}{i}{"|}"}","")
        return to_return,face_image

    def save(self):
        with open(self.dir+"\\memory.json","w",encoding="utf-8") as f:
            f.write(json.dumps(self.memory))

    def load(self):
        with open(self.dir+"\\memory.json","r",encoding="utf-8") as f:
            self.memory = json.loads(f.read())

    def get_direction(self,image_path):
        return get_move_direction.get_direction(image_path)

def request_llm(text_from_user, image=None,_name="test",_chat="0"):
    if characters.get(_chat)==None:
        characters[_chat] = Character(f"characters\\{_name}",_name)
    return characters[_chat].request_ready_chat(text_from_user,image)

def request_get_direction(image_path,_name="test",_chat="0"):
    if characters.get(_chat)==None:
        characters[_chat] = Character(f"characters\\{_name}",_name)
    return characters[_chat].request_ready_chat(image_path)

def request_llm_chat(text_from_user,_name="test"):
    if characters.get(_name)==None:
        characters[_name] = Character(f"characters\\{_name}",_name)
    return characters[_name].request(text_from_user)

import base64
import os
import time
from flask_cors import CORS
from flask import request, Flask, jsonify
app = Flask(__name__)
CORS(app)



@app.route('/ready_api/<name>/<chat>', methods=['POST'])
def ready_chat(name,chat):
    data = request.get_json()
    #print(data)
    image=None
    move="stop"
    base64_tts=""
    answer_llm=""
    face=""
    face_base64=""
    if data.get("image")!=None:
        image="input.jpg"
        with open("input.jpg", "wb") as f:
            f.write(base64.b64decode(data.get("image")))
    if data["message"].replace(" ","")!="":
        answer_llm=request_llm(data["message"],image,name,chat)
    else:
        move = request_get_direction(image,name,chat)
    if move==None:
        answer_llm = request_llm("system: ты видишь человека. ", image, name)
        move="stop"

        #answer_llm=answer_llm[0]
    emotions = {
        "удивлён": "surprise",
        "нормально": "normal",
        "злой": "evil",
        "скептический": "sceptical",
        "радость": "happy",
        "устал": "sleep",
        "хорошо": "good",
        "чего": "what",
    }
    if answer_llm!="":
        face = answer_llm[1]
        answer_llm=answer_llm[0]
        #print(face, answer_llm)
        with open("output_tts.wav", "rb") as f:
            base64_tts = base64.b64encode(f.read()).decode('ascii')
    emo="normal"
    if face!="":
        #[удивлён], [нормально], [злой], [скептический], [радость], [устал], [хорошо], [чего].

        if emotions.get(face)!=None:
            emo = emotions.get(face)

        with open(f"characters\\{name}\\face\\{emo}.png", "rb") as f:
            face_base64 = base64.b64encode(f.read()).decode('ascii')
    print(emo)
    return jsonify({"content": answer_llm,"move": move,"tts": base64_tts, "face": face_base64,"emotion":emo})

@app.route('/chat/<name2>', methods=['POST'])
def chat(name2):
    data = request.get_json()
    #print(data)
    image=None
    move="stop"
    base64_tts=""
    answer_llm=""
    face=""
    face_base64=""
    answer_llm = request_llm_chat(data["messages"], name2)

    face = answer_llm[1]
    answer_llm=answer_llm[0]
    if bool(system_config.get("use_llm").lower()=="true"):
        with open("output_tts.wav", "rb") as f:
            base64_tts = base64.b64encode(f.read()).decode('ascii')
    if face!="":
        emotions={
            "удивлён":"surprise",
            "нормально": "normal",
            "злой": "evil",
            "скептический": "sceptical",
            "радость": "happy",
            "устал": "sleep",
            "хорошо": "good",
            "чего": "what",
        }
        if emotions.get(face)==None:
            file = "normal"
        else:
            file = emotions.get(face)

        with open(f"characters\\{name2}\\face\\{file}.png", "rb") as f:
            face_base64 = base64.b64encode(f.read()).decode('ascii')
    return jsonify({"content": answer_llm,"tts": base64_tts, "face": face_base64})


if __name__=="__main__":
    print("App running time:",time.time() - frist_time)
    app.run(service_config.get("host"),int(service_config.get("port")),threaded=True)

#
#
# if __name__=="__main__":
#         print("start")
#         char = Character("characters\\guy","test")
#         while True:
#             inp=input()
#             frist_time = time.time()
#             print(char.request(inp,"test2.jpg"))
#             print(time.time()-frist_time)