import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY not found. Check that your .env file is in this folder "
                     "and contains: GROQ_API_KEY=your_key (no quotes, no spaces).")

client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

print("Models available to your key:\n")
for m in sorted(client.models.list().data, key=lambda x: x.id):
    print(" ", m.id)