from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-7b-base")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-7b-base")

input_text = "Translate to French: I love programming."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)