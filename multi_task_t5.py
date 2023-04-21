import torch
from torch import nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import copy


class MultiTaskT5(pl.LightningModule):
    def __init__(self, task_dict, tokenized_context, tokenizer=None):
        super().__init__()
        self.task_dict = task_dict
        self.num_tasks = len(task_dict)
        self.context = tokenized_context

        self.tokenizer = tokenizer if tokenizer else T5Tokenizer.from_pretrained('t5-small')

        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.task_decoders = nn.ModuleList([copy.deepcopy(self.model.decoder)
                                            for _ in range(self.num_tasks)])

    def forward(self, input_ids, labels, task_num):
        self.model.decoder = self.task_decoders[task_num]
        return self.model(input_ids=input_ids,
                          labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        task = batch['task'][0].item()

        out = self.forward(input_ids=input_ids,
                           labels=target_ids,
                           task_num=task)
        loss = out.loss
        self.log(f'{self.task_dict[task]}_train_loss', loss, on_epoch=True, on_step=True)

        if batch_idx % 1000 == 0:
            for i, t in enumerate(self.generate_example()):
                print(self.task_dict[i], ':\n', t, end='\n\n', sep='')
            sep = '#' * 60
            print(sep)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        task = batch['task'][0].item()

        out = self.forward(input_ids=input_ids,
                           labels=target_ids,
                           task_num=task)
        loss = out.loss
        self.log(f'{self.task_dict[task]}_val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-5)

    def generate(self, input_ids, attention_mask, task_num, max_length=64):
        self.model.decoder = self.task_decoders[task_num]
        generated_ids = self.model.generate(input_ids=input_ids.unsqueeze(0),
                                            attention_mask=attention_mask.unsqueeze(0),
                                            max_length=max_length)
        return generated_ids

    def generate_example(self):
        with torch.no_grad():
            input_ids = self.context['input_ids'][0]
            attention_mask = self.context['attention_mask'][0]

            res = []
            for tn in range(self.num_tasks):
                generated_ids = self.generate(input_ids, attention_mask, tn)
                generated_text = self.tokenizer.decode(generated_ids[0],
                                                       skip_special_tokens=True)
                res.append(generated_text)

        return res
