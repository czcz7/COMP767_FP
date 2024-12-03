import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True)

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)
# model.to(device)

prompt = "Answer this question in only one sentence, and make sure you include explaination: Is 7411 a prime number?"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

output = model.generate(
    input_ids, 
    # attention_mask=attention_mask,
    max_length=100,
    temperature=0.1,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("The response is: ")
print(response)