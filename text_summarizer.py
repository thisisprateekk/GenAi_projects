import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(input):
    output = pipe(input)
    return output[0]['summary_text']  # Assuming 'summary_text' is the correct key

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Paste your text here...", label="Input Text"),
    outputs=gr.Textbox(label="Summarized Output"),
    title=" prateek-genAI Text Summarizer",
    description="Enter a paragraph or article and get a concise summary using a text summarization model.",
    theme="default",  # You can try "compact" or "huggingface"
    examples=[
        ["The internet has transformed how we access information. With just a few clicks..."],
        ["Artificial Intelligence is a growing field in computer science that..."]
    ],
    allow_flagging="never"
)

demo.launch()
