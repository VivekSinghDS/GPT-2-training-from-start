import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
save_path = 'tokenized_data'
tokenizer = GPT2Tokenizer.from_pretrained(save_path)
from transformers import WEIGHTS_NAME, CONFIG_NAME
tokenizer.add_special_tokens({
    "eos_token":"</s>",
    "bos_token":"<s>",
    "unk_token":"<unk>",
    "pad_token":"<pad>",
    "mask_token":"<mask>"
})

config = GPT2Config(
    vocab_size = tokenizer.vocab_size,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id
)

model = TFGPT2LMHeadModel(config)
paths = [str(x) for x in Path("./text_data/").glob("**/*.txt")]
single_string = ''


for filename in paths:
    with open(filename, "r", encoding = 'utf-8') as f:
        x = f.read()

        single_string+=x+tokenizer.eos_token

string_tokenized = tokenizer.encode(single_string)
examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(string_tokenized) - block_size + 1, block_size):
  examples.append(string_tokenized[i:i + block_size])
inputs, labels = [], []
for ex in examples:
  inputs.append(ex[:-1])
  labels.append(ex[1:])
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])


num_epoch = 100
history = model.fit(dataset, epochs=num_epoch)

text = "I think"
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
print(beam_output)

import os
output_dir = './model_bn_custom_100/'
# creating directory if it is not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained(output_dir)
model_to_save.config.to_json_file(output_config_file)
# save tokenizer
tokenizer.save_pretrained(output_dir)