from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the saved model
model = AutoModelForCausalLM.from_pretrained("/storage/ice1/2/1/ysv3/saved").to("cuda")

modelName = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(modelName)

base_prompt_template: str = """
    You are a financial analyst designed to answer questions about business and finance. Your job is to reply to questions about finance topics and provide advice.

    Instruction: {query}
    
    Output:
"""
query = "What's the difference between a stock and an option?"
query = base_prompt_template.format(query=query)

modelInput = tokenizer(query, return_tensors="pt").to("cuda")

outputs = model.generate(**modelInput, max_new_tokens = 500, do_sample = True)
print(tokenizer.batch_decode(outputs)[0])
