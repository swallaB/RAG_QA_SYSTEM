from google import genai

client = genai.Client(api_key="AIzaSyAla8ZicqLnS125gN9UI8VTK30MU2-tfhs")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)