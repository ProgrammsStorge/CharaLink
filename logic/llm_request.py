import llama_cpp

llm=None
top_p=1
temperature=0.5
max_tokens=680
repeat_penalty=1.1
alternative_template=False
def init(model_path,n_ctx=8192,n_batch=3072,n_threads=12,n_gpu_layers=-1,top_p1=1,temperature1=0.5,max_tokens1=680,repeat_penalty1=1.1,alternative_template1=False):
    global top_p, temperature,max_tokens,repeat_penalty,llm,alternative_template
    alternative_template=alternative_template1
    top_p = top_p1
    temperature = temperature1
    max_tokens = max_tokens1
    repeat_penalty = repeat_penalty1
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads,
        verbose=False)

def request(chat):
    if alternative_template==False:
        return llm.create_chat_completion(
                messages=chat,
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=0.1,
                frequency_penalty=0.3,
                mirostat_tau=5.0,
                mirostat_eta=0.1,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=False
        )["choices"][0]["message"]["content"]
    else:
        chat_template = ""
        for message in chat:
            chat_template+=f"{message["role"]}: {message["content"]}\n"
        chat_template+="assistant:"
        return llm(
            chat_template,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=0.1,
            frequency_penalty=0.3,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["user:"],
            echo=False
        )["choices"][0]["text"]
