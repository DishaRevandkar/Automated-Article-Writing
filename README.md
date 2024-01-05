# Automated-Article-Writing: GPT-2 Language Model Fine-tuning and Text Generation

Let GPT2 write your technical publications and assignments!

Requirements

- Python 3.x
- PyTorch
- Transformers library by Hugging Face
- Other dependencies (specified in `requirements.txt`)

Text Generation
Load the trained model and tokenizer:

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("path/to/your/fine-tuned/model")
tokenizer = GPT2Tokenizer.from_pretrained("path/to/your/fine-tuned/tokenizer")

Use the writer_ai function for text generation:

from generate_text import writer_ai
writer_ai(model, device="cuda", tokenizer=tokenizer)
