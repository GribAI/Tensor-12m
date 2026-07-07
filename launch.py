import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
from prompt_toolkit import PromptSession
from prompt_toolkit.keys import Keys
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.widgets import Frame, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

class DummyTorchAudio:
    pass

sys.modules['torchaudio'] = DummyTorchAudio()

def get_model_files():
    return glob.glob("*.safetensors")

def select_file_with_prompt(files):
    if not files:
        print("No .safetensors files found.")
        return None
    if len(files) == 1:
        return files[0]
    selected = 0
    style = Style.from_dict({"selected": "fg:ansiblack bg:ansigreen"})
    kb = KeyBindings()

    @kb.add("up")
    def up(event):
        nonlocal selected
        selected = (selected - 1) % len(files)

    @kb.add("down")
    def down(event):
        nonlocal selected
        selected = (selected + 1) % len(files)

    @kb.add("enter")
    def enter(event):
        event.app.exit(result=files[selected])

    @kb.add("c-c")
    def cancel(event):
        event.app.exit(result=None)

    body = HSplit([])
    for idx, file in enumerate(files):
        if idx == selected:
            body.children.append(Window(Label(text=f"[x] {file}"), style="selected"))
        else:
            body.children.append(Window(Label(text=f"[ ] {file}")))

    container = Frame(body=body, title="Select model (up/down, enter)")
    layout = Layout(container)
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
    return app.run()

class ChatClient:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(".", local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(".", local_files_only=True)
        self.history = []

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    def chat(self):
        session = PromptSession()
        print("Chat with model. Type 'exit' to quit.")
        print("Type 'clear' to clear history.")
        while True:
            try:
                user_input = session.prompt("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.lower() == "clear":
                    self.history = []
                    print("History cleared.")
                    continue
                self.history.append("User: " + user_input)
                context = "\n".join(self.history[-10:])
                response = self.generate_response(context + "\nAssistant:")
                print("Assistant:", response)
                self.history.append("Assistant: " + response)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    files = get_model_files()
    selected_file = select_file_with_prompt(files)
    if selected_file:
        print(f"Loading {selected_file}...")
        client = ChatClient(selected_file)
        client.chat()
    else:
        print("No model selected.")