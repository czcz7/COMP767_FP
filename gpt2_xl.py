import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl", trust_remote_code=True)

json_file_path = "./Benchmark/primality_testing.json"
# json_file_path = "./Benchmark/graph_connectivity.json"
# json_file_path = "./Benchmark/senator_search.json"
# json_file_path = "./Benchmark/simple_graph_connectivity.json"
# json_file_path = "./Benchmark/simple_arithmetic.json"
with open(json_file_path, "r") as f:
    questions = json.load(f)

for temp in [0.6]:
    counter = 0
    results = []
    qid = 0

    for item in questions:
        print(f"########## Greedy Decoding : {counter} ##########")
        # print(f"########## {temp} : {counter} ##########")
        base_question = item["question"]
        number = item["number"]

        # question = f"{question} Directly give me the answer."
        # question = f"{item} Directly answer this question with explaination."
        question = f"499 is a prime number. 811 is a prime number. 500 is not a prime number. 928 is not a prime number. {base_question}"

        input_ids = tokenizer(question, return_tensors="pt").input_ids

        output = model.generate(
            input_ids, 
            # max_new_tokens=20,
            # max_length=30,
            max_length=50,
            # temperature=temp,
            # top_p=0.9,
            # do_sample=True,
            do_sample=False
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

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

    output_file_path = f"./Answers/gpt_xl_primality_greedy_fs.json"
    # output_file_path = f"./Answers/gpt_xl_arithmetic_greedy.json"
    # output_file_path = f"./Answers/gpt_xl_arithmetic_t{str(temp).replace('.', '')}.json"
    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4)

    print(f"Results for greedy decoding saved to {output_file_path}.")
    # print(f"Results for temperature {temp} saved to {output_file_path}.")

###

# prompt = "Is 14243 a prime number?"
# # prompt = "Current flight information (the following flights are one-way only, and all the flights available are included below):\nThere is a flight from city F to city L\nThere is a flight from city J to city E\nThere is a flight from city G to city B\nThere is a flight from city H to city K\nThere is a flight from city L to city M\nThere is a flight from city F to city H\nThere is a flight from city G to city J\nThere is a flight from city B to city I\nThere is a flight from city L to city A\nThere is a flight from city H to city N\nThere is a flight from city B to city D\nThere is a flight from city J to city C\n\nQuestion: Is there a series of flights that goes from city F to city I?"
# # prompt = "Was there ever a US senator that represented the state of Alabama and whose alma mater was MIT? Directly answer this question with explaination."

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# # attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

# output = model.generate(
#     input_ids, 
#     # attention_mask=attention_mask,
#     max_length=30,
#     # max_length=60,
#     # max_length=210,
#     temperature=0.1,
#     top_p=0.9,
#     do_sample=True,
# )

# response = tokenizer.decode(output[0], skip_special_tokens=True)

# print("The response is: ")
# print(response)