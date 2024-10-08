from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your raw text from the mental_health.txt file
with open('mental_health.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Define the splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the text into chunks
chunks = text_splitter.split_text(raw_text)

# Save the chunks into a new file
with open('mental_health_chunks.txt', 'w', encoding='utf-8') as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print(f"Number of chunks created: {len(chunks)}")