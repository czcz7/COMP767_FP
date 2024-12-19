import json

with open("./Benchmark/parsed_prime_questions.json", "r") as file:
    data = json.load(file)

for item in data:
    if "question" in item:
        item["question"] = item["question"].replace("the provided factor ", "")

with open("updated_prime_questions.json", "w") as file:
    json.dump(data, file, indent=4)

print("Updated JSON saved as 'updated_prime_questions.json'.")