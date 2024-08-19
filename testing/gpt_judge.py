import json

from dotenv import load_dotenv
from openai import OpenAI
import torch
import torchvision.datasets as dset
from tqdm import tqdm


load_dotenv()
client = OpenAI()

model = "gpt-4o-2024-08-06"
captions_file = "captions/val_224_no_merge.json"
out_file = "captions/llama_odd_one_out.json"

# Example
# captions = [
#     "A woman is standing in a living room with a fireplace.", # odd one out
#     'A woman stands in the dining area at the table.',
#     'A room with chairs, a table, and a woman in it.',
#     'A woman standing in a kitchen by a window',
#     'A person standing at a table in a room.',
#     'A living area with a television and a table'
# ]


def odd_one_out(captions):
    caption_message = "Which caption is the least relevant to the image?\n\n"
    caption_message += "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])

    response = client.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": "You are a summary assistant, skilled in summarizing posts and comments. Ensure the generated summary is concise and captures the essence of the content."},
            {"role": "system", "content": "You are a judge. You judge relevance of the captions to an image not available to you. You play the odd-one-out game. You are provided with multiple captions for a single image. You have to find the least relevant caption, which is not describing the same image as the other captions. Only answer with the number of the least relevant caption. Do not provide any other information, only the number itself."},
            {"role": "user", "content": caption_message}
        ],
        temperature=0,
        stream=False
    )

    return response.choices[0].message.content


ds = dset.CocoCaptions(
    root = '/home/xbuban1/coco/images/val2017',
    annFile = '/home/xbuban1/coco/annotations/captions_val2017.json'
)

with open(captions_file) as f:
    all_captions = json.load(f)

out = []

for i in tqdm(range(len(all_captions)), desc="Odd one out"):
    captions = ds[i][1]
    insert_index = torch.randint(0, len(captions), (1,)).item()
    captions = captions[:insert_index] + [all_captions[i]] + captions[insert_index:]

    response = odd_one_out(captions).strip().replace(".", "")
    correct = response == str(insert_index + 1)
    parse_issue = not response.isdigit()

    out.append({
        "index": i,
        "parse_issue": parse_issue,
        "correct": correct,
        "response": response,
        "insert_index": insert_index + 1,
        "captions": [f"{i+1}. {caption}" for i, caption in enumerate(captions)]
    })

with open(out_file, "w") as f:
    json.dump(out, f, indent=4)

issue_indices = [i for i, x in enumerate(out) if x["parse_issue"]]

print(f"{len(issue_indices)} Parse issues: {issue_indices}")
print(f"GPT Accuracy (lower is better): {sum(x['correct'] for x in out) / len(out)}")
print(f"Random accuracy: {1 / 6:.4f}")
