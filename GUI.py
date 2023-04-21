import tkinter as tk
from tkinter import ttk

import torch
from transformers import T5Tokenizer
from multi_task_t5 import MultiTaskT5


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

task_dict = {
    0: 'question_gen',
    1: 'sentence_comp',
    2: 'summarization'
}
tokenizer = torch.load('models/T5Tokenizer.pt',
                       map_location=device)
mtt5 = MultiTaskT5.load_from_checkpoint('models/model.bak',
                                        task_dict=task_dict,
                                        tokenizer=tokenizer,
                                        tokenized_context=None).to(device)
mtt5.eval()


def perform(task):
    text = input_text.get("1.0", "end-1c")
    length = int(spinbox.get())
    tok = tokenizer(text, padding='max_length',
                    max_length=512, truncation=True, return_tensors='pt')
    generated_ids = mtt5.generate(tok['input_ids'][0], tok['attention_mask'][0], task, max_length=length)
    res = tokenizer.decode(generated_ids[0],
                           skip_special_tokens=True)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"{res}")
    output_text.configure(fg="black")


def question_button_clicked():
    perform(0)


def sentence_completion_button_clicked():
    perform(1)


def summary_button_clicked():
    perform(2)


def on_input_focus_in(event):
    if input_text.get("1.0", "end-1c").strip() == "Enter input text here...":
        input_text.delete("1.0", tk.END)
        input_text.configure(fg="black")


def on_input_focus_out(event):
    if input_text.get("1.0", "end-1c").strip() == "":
        input_text.insert("1.0", "Enter input text here...")
        input_text.configure(fg="gray")


def on_output_focus_in(event):
    if output_text.get("1.0", "end-1c").strip() == "Output text will appear here...":
        output_text.delete("1.0", tk.END)
        output_text.configure(fg="black")


def on_output_focus_out(event):
    if output_text.get("1.0", "end-1c").strip() == "":
        output_text.insert("1.0", "Output text will appear here...")
        output_text.configure(fg="gray")


# Create the GUI window
window = tk.Tk()
window.title("MultiTaskT5")

# Set the window size to half the screen size while maintaining the same proportions
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = int(screen_width / 1.5)
window_height = int(window_width / 1.5)
window.geometry(f"{window_width}x{window_height}")

text_height, text_width = int(window_height * 0.02), int(window_width * 0.09)

# Set the style
style = ttk.Style(window)
style.configure("TButton", font=("Georgia", 20))
style.configure("TLabel", font=("Georgia", 20))
style.configure("TSpinbox", font=("Georgia", 16))

# Create an input text widget
input_text = tk.Text(window, height=text_height, width=text_width, font=("Georgia", 18))
input_text.pack(pady=10)
input_text.insert(tk.END, "Enter input text here...")  # Add default text as a placeholder
input_text.configure(fg="gray")
input_text.bind("<FocusIn>", on_input_focus_in)
input_text.bind("<FocusOut>", on_input_focus_out)

# Create a container for the Spinbox and buttons
container = ttk.Frame(window)
container.pack(pady=20)

# Create three buttons
question_button = ttk.Button(container, text="Question", command=question_button_clicked)
sentence_completion_button = ttk.Button(container, text="Sentence Completion",
                                        command=sentence_completion_button_clicked)
summary_button = ttk.Button(container, text="Summary", command=summary_button_clicked)

# Pack the buttons horizontally
question_button.pack(side="left", padx=10)
sentence_completion_button.pack(side="left", padx=10)
summary_button.pack(side="left", padx=10)

# Create a numeric spinbox
spinbox = ttk.Spinbox(container, from_=30, to=150, width=5, font=("Georgia", 16))
spinbox.pack(side="left", padx=10)
spinbox.set(50)  # Set the default value of the Spinbox to 50

# Create an output text widget
output_text = tk.Text(window, height=text_height, width=text_width, font=("Georgia", 18))
output_text.pack(pady=10)
output_text.insert(tk.END, "Output text will appear here...")  # Add default text as a placeholder
output_text.bind("<FocusIn>", on_output_focus_in)
output_text.bind("<FocusOut>", on_output_focus_out)
output_text.configure(fg="gray")
# output_text.config(state=tk.DISABLED)  # Disable editing of the output text widget

# Start the GUI event loop
window.mainloop()
