import torch.nn as nn 
from transformers import AutoModel
import torch
#custom_optimizer = torch.optim.AdamW([prefix_params] + list(classifier_params), lr=lr)
class BertMlpModel(nn.Module):
    '''
    BERT+MLP+Softmax
    '''
    def __init__(self, pretrained_model, num_labels):
        super(BertMlpModel, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.num_labels = num_labels
        #self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        # First hidden layer
        self.hidden1 = nn.Linear(self.model.config.hidden_size, 256)
        # Second hidden layer
        self.hidden2 = nn.Linear(256, 128)
        # Output layer
        self.output = nn.Linear(128, num_labels)
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidd = output.last_hidden_state
        
        #logits = self.classifier(last_hidd)
        x = self.relu(self.hidden1(last_hidd))
        # Pass through the second hidden layer, then apply ReLU
        x = self.relu(self.hidden2(x))
        # Pass through the output layer
        logits = self.output(x)
        return {"logits": logits}