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

def evaluate_model(model, dataloader, device):


    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:

        labels, input_ids, masks = batch['labels'], batch['input_ids'], batch['attention_mask']      
        labels=labels.to(device)
        input_ids=input_ids.to(device)
        masks=masks.to(device)

        # forward pass
        # name the output as `output`
        output = model(input_ids, attention_mask=masks)
        predictions = output['logits']


        predictions = torch.argmax(predictions, dim=-1)
        ## Flatten predictions and labels to match evaluation's format
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        dev_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)

    # compute and return metrics
    return dev_accuracy.compute()

def train(type,mymodel, train_dataloader, test_dataloader, lr, num_epochs, device):
    mymodel.to(device)
    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # we are only tuning parameters of MLP
    # BERT parameters are left frozen
    print(" >>>>>>>>  Initializing optimizer")
    if type=="head":
        
        non_bert_params = []
        # We assume that all parameters not belonging to `model.model` are non-BERT
        for name, param in mymodel.named_parameters():
            if not name.startswith('model.'):  # `model.model` refers to the BERT part
                non_bert_params.append(param)
        
        para=non_bert_params
        #mymodel.classifier.parameters()
    elif type=="full":
        para=mymodel.parameters()
    optimizer = torch.optim.AdamW(para, lr=lr)
    
    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    loss = torch.nn.CrossEntropyLoss().to(device)
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print("Epoch "+str(epoch + 1)+" training:")
        for i, batch in tqdm(enumerate(train_dataloader)):
            ## Get input_ids, attention_mask, labels
            labels, input_ids, masks = batch['labels'], batch['input_ids'], batch['attention_mask']
            ## Send to GPU
            labels=labels.to(device)
            input_ids=input_ids.to(device)
            masks=masks.to(device)
            ## Forward pass
            output = mymodel(input_ids, attention_mask=masks)
            predictions = output['logits']
            ## Cross-entropy loss
            centr = loss(predictions.view(-1, mymodel.num_labels), labels.view(-1))
            ## Backward pass
            centr.backward()
            ## Update Model
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ## Argmax get real predictions
            predictions = torch.argmax(predictions, dim=-1)
            ## Flatten predictions and labels to match evaluation's format
            predictions_flat = predictions.flatten()
            labels_flat = labels.flatten()
            # print("Shape of predictions:", predictions_flat.shape, predictions_flat.dtype)  
            # print("Shape of labels:", labels_flat.shape, labels_flat.dtype)  
            ## Update metrics
            train_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)
        # print evaluation metrics
        print(" ===> Epoch "+str(epoch + 1))
        train_acc = train_accuracy.compute()
        print(" - Average training metrics: accuracy="+str(train_acc))
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, test_dataloader, device)
        print(f" - Average validation metrics: accuracy="+str(val_accuracy))
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
        # test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        # print(f" - Average test metrics: accuracy={test_accuracy}")

        epoch_end_time = time.time()
        print("Epoch "+str(epoch + 1)+" took "+str(epoch_end_time - epoch_start_time)+" seconds")



if __name__=='__main__':
    ## Load dataset
    edudataset = load_dataset("json", data_files="./data/train.json")
    edudataset = edudataset['train']
    ## Preprocess datasets for NER task
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    edudataset = preprocess(dataset=edudataset,tokenizer=tokenizer)
    edudataset = edudataset.remove_columns(['full_text', 'document', 'tokens', 'trailing_whitespace'])
    ## Split dataset
    edudataset = edudataset.shuffle(seed=42)
    edudataset = edudataset.train_test_split(test_size=0.1)
    train_dataset = edudataset['train']
    test_dataset = edudataset['test']
    print('Train dataset:')
    print(train_dataset)
    print('Test dataset:')
    print(test_dataset)
    ## Create batches with Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    ## Create dataloaders with data collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    ## Train
    ## Print train info
    training_type="head"
    lr=0.001
    print("Training type is", training_type, ", Learning rate is", lr)
    ## Run Train 
    num_labels = 15 # 15 NER tags
    BertMLP = BertMlpModel(pretrained_model="bert-base-uncased", num_labels=num_labels)
    train(type=training_type,mymodel=BertMLP, train_dataloader=train_dataloader, test_dataloader=test_dataloader, lr=lr, num_epochs=10, device=device)

    
    


