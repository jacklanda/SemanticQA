# -*- coding: utf-8 -*-

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf")

# Let's chat for 5 lines
chat_history_ids = torch.tensor(
    tokenizer.encode("Daryl: Hello, I am Daryl. How are you?", add_special_tokens=True)
).unsqueeze(0)

for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors="pt"
    )
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if step > 0
        else new_user_input_ids
    )

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )

    # pretty print last ouput tokens from bot
    print(
        "Daryl: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )
