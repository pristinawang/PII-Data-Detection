from openai import OpenAI
from preprocess import Preprocess
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import torch
from transformers import get_scheduler
from model import BertMlpModel
import evaluate as evaluate
from tqdm import tqdm
import time
def preprocess(dataset, tokenizer):
    '''
    We only do this once and then we store the processed dataset
    One dataset at a time
    '''
    preprocess = Preprocess(dataset=dataset, tokenizer=tokenizer)
    tokenized_dataset = preprocess.preprocess()
    return tokenized_dataset

def get_dataset(train_size):
    ## Load dataset
    edudataset = load_dataset("json", data_files="./data/train.json")
    edudataset = edudataset['train']
    
    ## Split dataset
    edudataset = edudataset.shuffle(seed=42)
    edudataset = edudataset.train_test_split(test_size=0.1)
    train_dataset = edudataset['train'].select(range(train_size))
    test_dataset = edudataset['test']
    return train_dataset, test_dataset

def chat_gpt(message):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a useful assistant."},
        {"role": "user", "content": message}
    ]
    )

    return completion.choices[0].message.content

def get_fewshot_prompt(train_dataset, test_token, num_tok):
    examples=""
 
    for example in train_dataset:
        examples=examples+"Tokens:\n"+repr(','.join(example['tokens']))+"\n"
        examples=examples+"Labels:\n"+','.join(example['labels'])+"\n"
        examples=examples+"\n"

    pr='Please assign labels to the following list of tokens using BIO format to identify these 7 PII types. There are 15 labels. "O" is outside and rest of the labels represent beginning of a PII type or inside of a PII type. \
        For example, "B-NAME_STUDENT" is beginning of NAME_STUDENT type and "I-NAME_STUDENT" is inside of NAME_STUDENT type. \
        Use the following examples as reference. Answer in example format.\n \
        Labels: ["O", "B-NAME_STUDENT", "I-NAME_STUDENT", "B-EMAIL", "I-EMAIL", "B-USERNAME", "I-USERNAME", \
         "B-ID_NUM", "I-ID_NUM", "B-PHONE_NUM", "I-PHONE_NUM", "B-URL_PERSONAL", "I-URL_PERSONAL", \
         "B-STREET_ADDRESS", "I-STREET_ADDRESS"]\n \
        Seven PII Types:\n \
        NAME_STUDENT - The full or partial name of a student that is not necessarily the author of the essay. This excludes instructors, authors,\n \
        and other person names.\n \
        EMAIL - A student’s email address.\n \
        USERNAME - A student’s username on any platform.\n \
        ID_NUM - A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number.\n \
        PHONE_NUM - A phone number associated with a student.\n \
        URL_PERSONAL - A URL that might be used to identify a student.\n \
        STREET_ADDRESS - A full or partial street address that is associated with the student, such as their home address.\n \
        List of tokens:'+test_token+ '\n' \
        'Number of tokens that needed to be identified:'+str(num_tok)+'\n'+\
        'Examples: \n'+examples+'\n '

    
    
    return pr
        
def evaluate(labels, reply):
    print("__-------reply------")
    # s=s.split(':')[-1]
    # print(s.strip('\n'))
    reply=reply.split(':')[-1]
    reply=reply.strip('\n')
    print(reply)

    reply=reply.split(',')
    print("__-------reply------")
    acc=0
    for i in range(len(labels)):
        if i >len(reply)-1:
            break
        label=labels[i]
        if label==reply[i]:
            acc+=1
    print(acc/len(labels))
    return acc/len(labels)

if __name__=="__main__":
    test_size=10
    train_dataset, test_dataset = get_dataset(train_size=2)
    test_dataset = test_dataset.select(range(test_size))
    print("DATASETS:(train, test)")
    print(train_dataset, test_dataset)
    num_examples=test_size
    num_correct=0
    for test_example in test_dataset:
        num_tok=len(test_example['tokens'])
        test_token=repr(",".join(test_example['tokens']))
        prompt=examples = get_fewshot_prompt(train_dataset=train_dataset, test_token=test_token, num_tok=num_tok)
        reply=chat_gpt(prompt)
        
        num_correct=num_correct+evaluate(labels=test_example['labels'], reply=reply)
        
    print('Acc: ',num_correct/num_examples)

    
    
 