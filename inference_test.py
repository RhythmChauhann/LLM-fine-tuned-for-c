import sys
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("🐍 Python being used:", sys.executable)


CHECKPOINT_PATH = "./codet5_chunked/chunk_9"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"🚀 Loading model from: {CHECKPOINT_PATH}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
model.to(DEVICE)
model.eval()

print(f"✅ Model loaded successfully on {DEVICE.upper()}.")



confused_responses = [
    "Are you trying to confuse me? 🫠",
    "I have no idea what just happened. 🤯",
    "Uhh... what did I just read? 👀",
    "My circuits are overheating... try again? 🧠🔥",
    "That made less sense than a semicolon in an if-statement 😵",
    "Syntax error in my brain. Please rephrase. 💫",
    "I'm gonna pretend I didn’t see that. 🙈",
    "404 explanation not found. 🚫",
    "You broke me. Again. 😒",
    "Explain like I'm five, please. 🧸",
    "That's not even wrong. It's... something else. 🤡",
    "I need coffee for this one. ☕",
    "Huh? I'm just a broken AI, not a mind reader. 😩",
]


def main():
    print("\n🔥 AI Code Explainer is ready! Type your C++ code below.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("💀 Ask AI: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if not user_input.strip():
            print("⚠️ Empty input. Try again.")
            continue

       
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

       
        if len(explanation) < 3:
            explanation = random.choice(confused_responses)

        print("🤖 AI says:", explanation, "\n")

        
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(f"Input: {user_input}\nOutput: {explanation}\n\n")

if __name__ == "__main__":
    main()
