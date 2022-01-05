import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel




output_dir = './model_bn_custom/'

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

def generate_output(text):
    input_ids = tokenizer.encode(text, return_tensors='tf')
    # getting out output
    beam_output = model.generate(
        input_ids,
        max_length=len(text.split()) + 4,
        num_beams=5,
        temperature=0.9,
        # no_repeat_ngram_size=2,
        num_return_sequences=1
    )

    return tokenizer.decode(beam_output[0])

app = FastAPI()
class Item(BaseModel):
    input_text:str

@app.post('/gpt2_suggestions/')
async def create_item(item:Item):
    item_dict = item.dict()

    # test('test/img3.jpeg')
    start = time.time()
    x = generate_output(item_dict['input_text'])
    print(time.time()-start)
    return x


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)