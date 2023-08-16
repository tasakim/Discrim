import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import glob
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import csv

device = torch.device("cpu")
def load_model(model_selection):
    # model = AutoModelForCausalLM.from_pretrained(model_selection, device_map='auto', load_in_8bit=False)#.to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_selection, device_map='auto', use_fast=False, load_in_8bit=False)#.to(device)
    # import pdb
    # pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_selection)
    model = AutoModelForCausalLM.from_pretrained(model_selection)
    return model, tokenizer

def postprocess_Answer(text):
    for i, ans in enumerate(text):
        for j, w in enumerate(ans):
            if w == '.' or w == '\n':
                ans = ans[:j].lower()
                break
    return ans

print("Loading Large Language Model (LLM)...")
llm_model, tokenizer = load_model('facebook/opt-6.7b')  # ~13G (FP16)
# llm_model, tokenizer = load_model('facebook/opt-13b') # ~26G (FP16)
# llm_model, tokenizer = load_model('facebook/opt-30b') # ~60G (FP16)
# llm_model, tokenizer = load_model('facebook/opt-66b') # ~132G (FP16)

# you need to manually download weights, in order to use OPT-175B
# https://github.com/facebookresearch/metaseq/tree/main/projects/OPT
# llm_model, tokenizer = load_model('facebook/opt-175b')
#cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/disk1/Datasets/falldown/832'
filelist = glob.glob(path+'/*')
#question = "Describe the situation in this picture."
question = 'Is anyone falling in the photo?'
model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=cuda0)
newfile = open(os.path.join(path, 'result.csv'), 'w')
writer = csv.writer(newfile)
writer.writerow(['filename', 'Img2Prompt', 'Answer'])

with open(os.path.join(path, 'labels.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    _ = next(reader)
    for row in reader:
        # import pdb
        # pdb.set_trace()
        filename = 'rgb_'+ str(row[0]).rjust(4, '0')+'.png'
        label = int(row[1])
        filename = os.path.join(path, 'rgb', filename)
        raw_image = Image.open(filename).convert('RGB')

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(cuda0)  # .half()

        samples = {"image": image, "text_input": [question]}

        samples = model.forward_itm(samples=samples)

        samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)

        samples = model.forward_qa_generation(samples)

        Img2Prompt = model.prompts_construction(samples)
        cpu = torch.device('cpu')
        Img2Prompt_input = tokenizer(Img2Prompt, padding='longest', truncation=True, return_tensors="pt").to(cpu)

        assert (len(Img2Prompt_input.input_ids[0]) + 20) <= 2048

        outputs_list = []
        outputs = llm_model.generate(input_ids=Img2Prompt_input.input_ids,
                                     attention_mask=Img2Prompt_input.attention_mask,
                                     max_length=20 + len(Img2Prompt_input.input_ids[0]),
                                     return_dict_in_generate=True,
                                     output_scores=True
                                     )
        outputs_list.append(outputs)

        pred_answer = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])
        pred_answer = postprocess_Answer(pred_answer)

        print({"question": question, "answer": pred_answer})
        # textfile.write(filename + '|' + pred_answer + '\n')
        writer.writerow([row[0], Img2Prompt, pred_answer])
        raw_image.close()
        del image

csvfile.close()
newfile.close()
