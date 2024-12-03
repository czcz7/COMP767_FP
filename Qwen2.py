import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
model.to(device)

json_file_path = "./Benchmark/primality_testing.json"
with open(json_file_path, "r") as f:
    questions = json.load(f)

counter = 0
results = []

for item in questions:
    print("########## " + str(counter) + " ##########")
    base_question = item["question"]
    number = item["number"]
    question = f"Answer this question in only one sentence, and make sure you include explaination.  {base_question}"
    
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=100,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"{number}: {response}")
    counter += 1

    results.append({
        "number": number,
        "answer": response
    })

output_file_path = "./Answers/qwen_1.5_primality_t09.json"
with open(output_file_path, "w") as output_file:
    json.dump(results, output_file, indent=4)

print(f"All responses have been saved to {output_file_path}.")
