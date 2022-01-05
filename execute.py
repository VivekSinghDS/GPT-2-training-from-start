import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
import time
start = time.time()
output_dir = './model_bn_custom/'

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

text = "The update now draws"
print(text.split())
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length = len(text.split())+4,
  num_beams = 5,
  temperature = 0.9,
  # no_repeat_ngram_size=2,
  num_return_sequences=1
)

print(tokenizer.decode(beam_output[0]))
print(time.time()-start)