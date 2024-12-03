from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "baichuan-inc/Baichuan-7B", 
    device_map="auto", 
    trust_remote_code=True,
    offload_folder="./offload"
)

model = model.to("mps")

prompt = "Is 7411 a prime number? Firstly you should answer yes or no. If your answer is no, on the second line give me an explanation in one sentence."

# Tokenize input and move tensors to MPS
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to("mps") for key, value in inputs.items()}

# Generate response
pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)

# Decode and print the response
print(tokenizer.decode(pred[0], skip_special_tokens=True))