import uvicorn
from fastapi import FastAPI
from chatbot import ChatBot

app = FastAPI()

@app.get('/{message}')
async def chat(message: str):
    answer = chatbot.get_msg(message)
    return {'Bot:': answer}


if __name__ == '__main__':
    chatbot = ChatBot()
    uvicorn.run(app, host="0.0.0.0", port=8000)

