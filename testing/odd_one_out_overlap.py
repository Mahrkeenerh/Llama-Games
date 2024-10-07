import json


with open("captions/ved_odd_one_out.json") as f:
    ved = json.load(f)

with open("captions/3.1_val_224_merge_odd_one_out.json") as f:
    llama = json.load(f)

total_correct = 0
both_correct = 0

for i in range(len(ved)):
    if ved[i]["correct"] or llama[i]["correct"]:
        total_correct += 1
    
    if ved[i]["correct"] and llama[i]["correct"]:
        both_correct += 1

ved_correct = sum(x["correct"] for x in ved)
llama_correct = sum(x["correct"] for x in llama)

print(f"Total correct: {total_correct}")
print(f"Both correct: {both_correct}")
print(f"Llama overlap: {both_correct / llama_correct}")
print(f"Total overlap: {both_correct / total_correct}")
print(f"VED correct: {ved_correct}")
print(f"Llama correct: {llama_correct}")
