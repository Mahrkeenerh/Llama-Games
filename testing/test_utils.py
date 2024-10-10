import json

import evaluate
from IPython.display import display
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from modeling import *


def caption_coco(ds, model, device, out_name):
    with torch.no_grad():
        outs = []

        for img, _ in tqdm(ds, desc="Generating captions", total=len(ds)):
            img = transforms.PILToTensor()(img).to(device)

            if not isinstance(model, VEDModel):
                img = model.vit_processor(img.unsqueeze(0), return_tensors='pt').pixel_values
                img = img.unsqueeze(0).to(device)

            out = model.generate(img, max_new_tokens=64, do_sample=False, top_p=None, temperature=None)
            outs.append(out)

        with open(out_name, 'w') as f:
            json.dump(outs, f, indent=4)


def caption_app(dl, model, out_name):
    with torch.no_grad():
        outs = []

        for data in tqdm(dl, desc="Generating captions", total=len(dl)):
            image_batches, [hints, descriptions] = data

            out = model.generate(
                image_batches,
                hints=hints,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repetition_penalty=1.1
            )

            outs.append(out)

        with open(out_name, 'w') as f:
            json.dump(outs, f, indent=4)


bleu = None
meteor = None
rouge = None
init_done = False

def init_eval():
    global bleu, meteor, rouge, init_done

    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    init_done = True


def score_single(out, ann, do_print=False):
    if not init_done:
        init_eval()

    references = [[x] for x in ann]
    predictions = [out]

    bleu_score = bleu.compute(predictions=predictions, references=[references])
    meteor_score = meteor.compute(predictions=predictions, references=[references])
    rouge_score = rouge.compute(predictions=predictions, references=[references])

    if do_print:
        print(bleu_score)
        print(meteor_score)
        print(rouge_score)

    return {**bleu_score, **meteor_score, **rouge_score}


def app_single(llama_model, ds, index, device, preview_image_count=1):
    if index == -1:
        index = torch.randint(0, len(ds), (1,)).item()

    imgs, [hint, description] = ds[index]

    # Only works in Jupyter
    try:
        display_imgs = [transforms.functional.to_pil_image(img) for img in imgs][:preview_image_count]
        display(*display_imgs)
    except NameError:
        pass

    print(f'Index: {index}\n')
    print(f"{hint}{description}")
    print("\n" + "-" * 80)

    if hasattr(llama_model, 'vit_processor'):
        img_in = llama_model.vit_processor(imgs, return_tensors='pt').pixel_values
        img_batch = img_in.unsqueeze(1).to(device)
    else:
        img_batch = None

    hint_tokens = llama_model.tokenizer(hint, return_tensors='pt').input_ids[0].to(device).unsqueeze(0)

    out = llama_model.generate(
        img_batch,
        hints=hint_tokens,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.1,
        repetition_penalty=1.1
    )

    print(f"{hint}{out}")


def coco_single_compare(llama_model, ved_model, ds, index, device):
    if index == -1:
        index = torch.randint(0, len(ds), (1,)).item()

    img, ann = ds[index]

    # Only works in Jupyter
    try:
        display(img)
    except NameError:
        pass

    print(f'Index: {index}\n')
    print("Target annotations:")
    print("\n".join(ann))
    print("\nOutput annotations:")

    img = transforms.PILToTensor()(img).to(device)
    img_in = llama_model.vit_processor(img.unsqueeze(0), return_tensors="pt").pixel_values
    img_batch = img_in.unsqueeze(0).to(device)
    out = llama_model.generate(img_batch, max_new_tokens=64, do_sample=False, top_p=None, temperature=None)

    print("Llama:")
    print(out)
    score_single(out, ann, do_print=True)

    out = ved_model.generate(img, max_new_tokens=64)

    print("\nVED:")
    print(out)
    score_single(out, ann, do_print=True)


def coco_eval_scores(ds, out_name):
    with open(out_name) as f:
        outs = json.load(f)

    # replace " " with "None" in the annotations
    outs = [x if x.strip() != "" else "None" for x in outs]

    scores = []

    for out, (_, ann) in tqdm(zip(outs, ds), desc="Calculating scores", total=len(ds)):
        scores.append(score_single(out, ann))

    return scores


def coco_eval_avg(ds, out_name):
    scores = coco_eval_scores(ds, out_name)
    len_scores = len(scores)

    avg = {k: sum(x[k] for x in scores) / len_scores for k in ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL']}

    return avg


def get_similarities(model, ds, captions):
    similarities = []

    for i in tqdm(range(0, len(captions)), desc="Calculating similarities"):
        caption = captions[i]
        caption_embedding = model.encode(caption)
        annotation_embeddings = model.encode(ds[i][1])

        similarities.append(model.similarity(caption_embedding, annotation_embeddings).squeeze())

    return similarities


def get_app_similarities(model, target, source):
    similarities = []

    for i in tqdm(range(0, len(target)), desc="Calculating similarities"):
        target_embedding = model.encode(target[i])
        source_embeddings = model.encode(source[i])

        similarities.append(model.similarity(target_embedding, source_embeddings).squeeze())

    return similarities


def get_random_app_similarities(model, target):
    similarities = []

    for i in tqdm(range(0, len(target)), desc="Calculating similarities"):
        index = torch.randint(0, len(target), (1,)).item()
        while index == i:
            index = torch.randint(0, len(target), (1,)).item()

        target_embedding = model.encode(target[i])
        source_embedding = model.encode(target[index])

        similarities.append(model.similarity(target_embedding, source_embedding).squeeze())

    return similarities


def get_empty_similarities(model, target):
    similarities = []

    for i in tqdm(range(0, len(target)), desc="Calculating similarities"):
        target_embedding = model.encode(target[i])
        source_embedding = model.encode("")

        similarities.append(model.similarity(target_embedding, source_embedding).squeeze())

    return similarities


def get_similarities_avg(model, ds, captions):
    similarities = get_similarities(model, ds, captions)

    avg = torch.cat(similarities).mean(dim=0)

    return avg


def get_intra_similarities(model, ds):
    similarities = []

    for img, ann in tqdm(ds, desc="Calculating intra-similarities"):
        annotation_embeddings = model.encode(ann)

        s = model.similarity(annotation_embeddings, annotation_embeddings)
        # Take the upper triangular part of the similarity matrix - over the diagonal
        similarities.append(torch.cat([s[i][i+1:] for i in range(s.shape[0]-1)]))

    return similarities


def get_intra_similarities_avg(model, ds):
    similarities = get_intra_similarities(model, ds)

    avg = torch.cat(similarities).mean(dim=0)

    return avg


def get_intra_dissimilarities(model, ds):
    dissimilarities = []

    for i, (img, ann) in tqdm(enumerate(ds), desc="Calculating intra-dissimilarities", total=len(ds)):
        index = torch.randint(0, len(ds), (1,)).item()
        while index == i:
            index = torch.randint(0, len(ds), (1,)).item()

        ann_dis = ds[index][1]

        annotation_embeddings = model.encode(ann)
        dis_annotation_embeddings = model.encode(ann_dis)

        s = model.similarity(annotation_embeddings, dis_annotation_embeddings)
        # Take the upper triangular part of the similarity matrix - over the diagonal
        dissimilarities.append(torch.cat([s[i][i+1:] for i in range(s.shape[0]-1)]))

    return dissimilarities


def get_intra_dissimilarities_avg(model, ds):
    dissimilarities = get_intra_dissimilarities(model, ds)

    avg = torch.cat(dissimilarities).mean(dim=0)

    return avg


def normalize_scores(score_dict):
    keys = score_dict['ved'].keys()
    max_values = {k: max([score_dict[model][k] for model in score_dict.keys()]) for k in keys}
    min_values = {k: min([score_dict[model][k] for model in score_dict.keys()]) for k in keys}
    
    normalized_scores = {
        model: {k: (score_dict[model][k] - min_values[k]) / (max_values[k] - min_values[k])
        for k in keys} for model in score_dict.keys()
    }
    return normalized_scores


def lineplot_scores(score_dict, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Score type')
    plt.ylabel('Score value')
    plt.xticks(rotation=45)

    for model, scores in score_dict.items():
        plt.plot(scores.keys(), scores.values(), label=model, marker='.')

    plt.legend()
    plt.show()


def boxplot_similarities(
    similarities,
    avg_sims,
    names
):
    fig, ax = plt.subplots()
    ax.boxplot(similarities, showfliers=False)
    # add means
    ax.plot(range(1, len(similarities) + 1), avg_sims, 'ro')
    ax.set_xticklabels(names)
    plt.xticks(rotation=30)
    plt.show()
