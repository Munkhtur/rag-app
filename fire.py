from fireworks.client import Fireworks

client = Fireworks(api_key="fw_3ZX4yv1JWAqk86NGtaZz2v7G")
response = client.chat.completions.create(
model="accounts/gmunkhtur-df7f59/models/mymodel",
messages=[{
   "role": "user",
   "content": "Say this is a test",
}],
)

print(response.choices[0].message.content)
