import os
import numpy as np
import argparse
import replicate
from utility_fns import chunk_text
from utility_fns import cosine_similarity
import pickle
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Usage: python create_chunk_embeddings.py <directory> <chunk_size> <query>

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings(text_batch):
    try:
        output = replicate.run(
            "shelbyt/all-mpnet-base-v2-a40:6bb0a786ca129e84831127c496a639db837378750c0a268551fc9dedbbfb3a0e",
            input={"text_batch": json.dumps(text_batch)},
        )
        # Check if output is not empty and has the expected structure
        if output and isinstance(output, list) and "embedding" in output[0]:
            return output
        else:
            print(f"Unexpected output from replicate.run: {output}")
            return None
    except Exception as e:
        print(f"Error occurred while getting embeddings: {e}")
        return None


def get_embeddings_query(text):
    try:
        output = replicate.run(
            "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
            input={"text": text},
        )
        # Check if output is not empty and has the expected structure
        if output and isinstance(output, list) and "embedding" in output[0]:
            return output[0]["embedding"]  # return only the embedding array
        else:
            print(f"Unexpected output from replicate.run: {output}")
            return None
    except Exception as e:
        print(f"Error occurred while getting embeddings: {e}")
        return None


def process_files_in_directory(directory_path, chunk_size):
    global vdb
    vdb = []
    for filename in os.listdir(directory_path):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Processing ", filename)
        print("Current Time =", current_time)
        embeddings_file = f"{os.path.splitext(filename)[0]}_{chunk_size}.pkl"
        if not os.path.exists(embeddings_file):
            with open(os.path.join(directory_path, filename), "r") as file:
                text = file.read()
                chunks = chunk_text(text, chunk_size)
                embd = get_embeddings(chunks)
            if embd:
                # Save embeddings and chunks to file
                with open(embeddings_file, "wb") as f:
                    pickle.dump(
                        [
                            {"chunk": c, "embedding": e["embedding"]}
                            for c, e in zip(chunks, embd)
                        ],
                        f,
                    )
                print(f"Embeddings and chunks saved to {embeddings_file}")
            else:
                print(f"Skipping file due to error: {filename}")
        # Load embeddings and chunks from file
        try:
            with open(embeddings_file, "rb") as f:
                embd = pickle.load(f)
            print(f"Embeddings and chunks loaded from {embeddings_file}")
            vdb.extend(embd)
        except:
            print("Embeddings file was not created or found")
        # Append the embeddings and chunks to vdb


def generate_sentence(prompt, columns=None):
    model = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )
    return {
        "prompt": prompt,
        "response": response.choices[0].message.content,
        "model": model,
    }


def run_prompt(query, chunks):
    q_embd = get_embeddings_query(query)
    ratings = [cosine_similarity(q_embd, x["embedding"]) for x in vdb]
    k = 4
    idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted
    # prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
    # Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[vdb[index]['chunk'] for index in idx]}"
    prompt = f"""Imagine you are an expert handyman and teacher providing essential tips and advice on home improvement 
                    and repair tasks specifically tailored for amateur homeowners and apartment dwellers. Your responses 
                    should include not only the necessary steps to complete each task but also estimated costs and timelines. 
                    Your advice should be clear and easy to follow for someone with minimal DIY experience. 
                Question: {query}
                Relevant Information: {[vdb[index]['chunk'] for index in idx]}
                Provide straightforward, step-by-step guidance that accounts for the user’s limited experience in handling hardware and home repairs."""

    print(prompt)
    result = generate_sentence(prompt)
    print(result["response"])


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("dir", type=str, help="The directory of files to process")
    parser.add_argument("chunk_size", type=int, help="The size of chunks")
    parser.add_argument("query", type=str, help="Prompt to run")

    args = parser.parse_args()

    chunks = process_files_in_directory(args.dir, args.chunk_size)
    # print("Asking Model: ", args.query)
    # print()
    print("Response: ")
    print()
    run_prompt(args.query, chunks)


if __name__ == "__main__":
    main()
