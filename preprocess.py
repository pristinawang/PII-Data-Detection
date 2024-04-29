from datasets import ClassLabel
import logging
class Preprocess:
    '''
    data fomatting: dataset has different tokenization from designated model design
    this class makes dataset's labels fit the designated tokenizer
    https://towardsdatascience.com/named-entity-recognition-with-deep-learning-bert-the-essential-guide-274c6965e2d
    '''
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer=tokenizer
    
    def preprocess(self):
        '''
        Preprocess method: label all tokens of a word (i.e. propagate the label on all tokens)
        https://discuss.huggingface.co/t/how-to-deal-with-differences-between-conll-2003-dataset-tokenisation-and-ber-tokeniser-when-fine-tuning-ner-model/11129/2
        tokenizing pretokenized input words: https://huggingface.co/learn/nlp-course/en/chapter7/2
        '''

        classmap = ClassLabel(num_classes=15, names=["O", "B-NAME_STUDENT", "I-NAME_STUDENT", "B-EMAIL", "I-EMAIL", "B-USERNAME", "I-USERNAME", 
         "B-ID_NUM", "I-ID_NUM", "B-PHONE_NUM", "I-PHONE_NUM", "B-URL_PERSONAL", "I-URL_PERSONAL", 
         "B-STREET_ADDRESS", "I-STREET_ADDRESS"])
        self.dataset = self.dataset.map(lambda y: {"labels": classmap.str2int(y["labels"])})
        tokenized_dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True)

        return tokenized_dataset
        

        
    
    def tokenize_and_align_labels(self,examples):
        '''
        Only label the first token of a given word.
        '''
    
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) # padding="max_length", max_length=512
       
        labels = []

        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                    
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
           
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
  
        return tokenized_inputs


