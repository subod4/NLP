from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import zipfile

# Step 1: Extract the model (if not already extracted)
zip_file_path = "saved_model.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("./saved_model")

# Step 2: Load the model and tokenizer
model_path = "./saved_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 3: Define the summarization function
def summarize_text(text):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=10,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Step 4: Test the model
input_text = "Your input text here..."
summary = summarize_text(input_text)
print("Summary:", summary)

# Optional: Create a Gradio interface
import gradio as gr
interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(label="Input Text", lines=10, placeholder="Enter text here..."),
    outputs=gr.Textbox(label="Summary", lines=5),
    title="Fine-Tuned T5 Summarizer",
    description="This summarizer uses a fine-tuned T5 model for text summarization.",
)
interface.launch()