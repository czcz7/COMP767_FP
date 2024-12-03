# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large", trust_remote_code=True)

prompt = "Is 14243 a prime number?"
# prompt = "Current flight information (the following flights are one-way only, and all the flights available are included below):\nThere is a flight from city F to city L\nThere is a flight from city J to city E\nThere is a flight from city G to city B\nThere is a flight from city H to city K\nThere is a flight from city L to city M\nThere is a flight from city F to city H\nThere is a flight from city G to city J\nThere is a flight from city B to city I\nThere is a flight from city L to city A\nThere is a flight from city H to city N\nThere is a flight from city B to city D\nThere is a flight from city J to city C\n\nQuestion: Is there a series of flights that goes from city F to city I?"

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