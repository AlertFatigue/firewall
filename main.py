import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key from environment asdfasdfasdfsadf
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your alert prompt
prompt = """
Alert details:
- VLAN: Guest
- Device: IoT camera
- Alert frequency: 10 times today
- Protocol: HTTP

Task:
Classify this alert as:
1. High priority threat
2. Likely noise

Explain briefly.
"""

# Send request to LLM
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a cybersecurity analyst helping with alert triage."},
        {"role": "user", "content": prompt}
    ]
)

# Print result
print(response.choices[0].message.content)
