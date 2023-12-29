# Input: CSV file
# Operations:
# 1. Run each row through gpt to create a sentence (OAI GPT4 API)
# 2. Convert each sentence to an embedding vector (1536) (OAI ADA-EMBED API)
# 3. Batch Upsert the embedding vector into Pinecone using the index, vector, and column info as metadata (Pinecone API)

import os
import pinecone
import csv
from openai import OpenAI
from openai import AsyncOpenAI
import requests
from dotenv import load_dotenv
import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize APIs
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-west1-gcp')
# Function to generate sentence using GPT-4
def generate_sentence(input_row, columns=None):
    prompt = f"convert this row from a dataset into a 100 word concise but descriptive paragraph with all the  technical specs that I can convert into an embedding. here are the columns for the dataset ensure information from each column is included: {columns} -> {input_row}"
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=3000)
    return response.choices[0].message.content
    # return response.choices[0].text.strip()

# Function to convert sentence to embedding
def convert_to_embedding(sentence):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=sentence
    ) 
    return response.data[0].embedding

# Function to upsert into Pinecone
def upsert_to_pinecone(index, vector, metadata):
    pinecone_client = pinecone.Index(index)
    pinecone_client.upsert(items=[(index, vector, metadata)])


def main():
    parser = argparse.ArgumentParser(description='Process a CSV file.')
    parser.add_argument('csvfile', type=str, help='The CSV file to process')
    args = parser.parse_args()

    with open(args.csvfile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = " ".join(reader.fieldnames)  # Get headers as a space-separated sentence
        for row in reader:
            row_str = " ".join([str(val) for val in row.values()])
            generated_sentence = generate_sentence(row_str, columns)
            #print (generated_sentence)
            embedding_vector = convert_to_embedding(generated_sentence)
            print (embedding_vector)
            exit(1)
            metadata = {key: row[key] for key in row if key != 'text'}
            print (metadata)
            print (generate_sentence)
            print (metadata)
            # upsert_to_pinecone(row['index'], embedding_vector, metadata)

if __name__ == "__main__":
    main()



