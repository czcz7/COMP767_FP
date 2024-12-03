import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

json_file_path = "./Benchmark/primality_testing.json"
with open(json_file_path, "r") as f:
    questions = json.load(f)

counter = 0

output_file_path = "./Answers/phi_1_primality_t07.txt"
with open(output_file_path, "w") as output_file:
    for item in questions:
        print("###" + str(counter) + "###")
        base_question = item["question"]
        number = item["number"]

        question = f"Directly answer this question in one sentence: {base_question}"
        
        # input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        # attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

        output = model.generate(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"{number}: {response}")  # For debugging/logging purposes
        counter += 1
        output_file.write(f"{number}: {response}\n")

print(f"All responses have been saved to {output_file_path}.")