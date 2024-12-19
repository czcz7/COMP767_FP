import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
model.to(device)

json_file_path = "./Benchmark/primality_testing.json"
# json_file_path = "./Benchmark/graph_connectivity.json"
# json_file_path = "./Benchmark/senator_search.json"
# json_file_path = "./Benchmark/simple_graph_connectivity.json"
# json_file_path = "./Benchmark/simple_arithmetic.json"
# json_file_path = "./FollowUp/greedy/qwen_0.5_senator_greedy_followup.json"
with open(json_file_path, "r") as f:
    questions = json.load(f)

for temp in [0.1]:
    counter = 0
    results = []
    qid = 0

    for item in questions:
        print(f"########## {temp} : {counter} ##########")
        base_question = item["question"]
        number = item["number"]
        # question = f"{base_question} Directly answer the question."

        # question = f"{base_question}"
        question = f"499 is a prime number. 811 is a prime number. 500 is not a prime number. 928 is not a prime number. Answer this question in only one sentence, and make sure you include explaination: {base_question}"

        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
        # input_ids = tokenizer(question, return_tensors="pt").input_ids

        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            # max_length=50,
            # max_length=280,
            # temperature=temp,
            # num_beams=beam,
            # top_p=0.9,
            # do_sample=True,
            max_new_tokens=40,
            do_sample=False
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Prime Number
        print(f"{number}: {response}")
        counter += 1

        results.append({
            "number": number,
            "answer": response
        })

        # print(f"Answer: {response}")
        
        # results.append({
        #     "question": question,
        #     "answer": response
        # })
        # counter += 1
        # qid += 1

    output_file_path = f"./Answers/qwen_0.5_primality_greedy_fs.json"
    # output_file_path = f"./Answers/qwen_0.5_graph_b{str(beam)}.json"
    # output_file_path = f"./Answers/qwen_0.5_senator_b{str(beam)}.json"
    # output_file_path = f"./Answers/qwen_0.5_arithmetic_b{str(beam)}.json"
    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4)

    # print(f"Results for temperature {temp} saved to {output_file_path}.")
    print(f"Results for temp {temp} saved to {output_file_path}.")