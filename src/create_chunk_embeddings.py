import os
import numpy as np
import argparse
import replicate
from utility_fns import chunk_text
from utility_fns import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Usage: python create_chunk_embeddings.py <directory> <chunk_size> <query>

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
vdb = []  # vector database


def get_embeddings(text):
    output = replicate.run(
        "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
        input={"text": text},
    )
    return output[0]["embedding"]  # return only the embedding array

def process_files_in_directory(directory_path, chunk_size):
    for filename in os.listdir(directory_path):
        with open(os.path.join(directory_path, filename), "r") as file:
            text = file.read()
            chunks = chunk_text(text, chunk_size)
            for chunk in chunks:
                embd = get_embeddings(chunk)
                vdb.append(embd)
                # print("Embedding created and loaded")
    return chunks


def generate_sentence(prompt, columns=None):
    model = "gpt-3.5-turbo"
    response = client.chat.completions.create(
    model=model,
        messages=[
        {"role": "user", "content": prompt},
    ],
    max_tokens=3000)
    return {"prompt": prompt, "response": response.choices[0].message.content, "model": model}

def run_prompt(query, chunks):
    q_embd = get_embeddings(query)
    ratings = [cosine_similarity(q_embd, x) for x in vdb]
    k = 4
    idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted
    prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
    Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[chunks[index] for index in idx]}"
    result = generate_sentence(prompt)
    print (result["response"])


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("dir", type=str, help="The directory of files to process")
    parser.add_argument("chunk_size", type=int, help="The size of chunks")
    parser.add_argument("query", type=str, help="Prompt to run")

    args = parser.parse_args()

    # print(process_files_in_directory(args.dir, args.chunk_size))
    chunks = process_files_in_directory(args.dir, args.chunk_size)
    print("Asking Model: ", args.query)
    print()
    print("Response: ")
    print()
    run_prompt(args.query, chunks)


if __name__ == "__main__":
    main()
