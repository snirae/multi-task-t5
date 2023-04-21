# multi-task-t5
Training modified T5 model to perform several tasks at once

This project has started with the goal of fine-tuning a language model to generate questions from a given text.
Eventually, we decided to expand the capability to generating sentence completion type of questions and summarization.
We used a method called multitask learning to create a model that can, theoretically, perform
any given number of tasks that it's trained on.

### `multi_task_t5.py`
This file contains the model class itself, which consists of an encoder, that's shared for all the tasks, and several decoders,
one for each task.
Recent studies show improved results with this approach, while also reducing model size compared to architectures
where there is an encoder for each task separately.

To generate output from an input text, you should tokenize the input using T5Tokenizer and pass the input_ids,
attention_mask and task number. Then, decode the generated ids with the tokenizer.

### `questiongeneration.ipynb`
Contains the datasets and dataloader creation (using SQuAD, Race and Billsum datasets), as well
as the training loop, to recreate the trained model. We used Kaggle's environment and GPUs.

### `GUI.py`
Once you got the model (that you trained) and the tokenizer (from HuggingFace🤗), you can run the GUI in the file for
more convenient use.

![example](https://user-images.githubusercontent.com/110405826/233517119-5496e6ea-0ea9-4b8c-a8a9-d31fcb846d6d.gif)
