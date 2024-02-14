import argparse
import os
import re
from tika import parser

def process_file(file_path, txt_path):
    # Parse the file
    parsed = parser.from_file(file_path)

    # The parsed content is in the 'content' key of the returned dictionary
    content = parsed['content']

    if content is None:
        print(f"Failed to extract text from {file_path}")
        return

    if not isinstance(content, str):
        content = str(content)

    # Remove extra newlines and empty spaces
    content = re.sub(r'\n\s*\n', '\n', content)
    content = re.sub(r' +', ' ', content)

    # Write the content to a txt file
    with open(txt_path, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='Convert files to TXT.')
    parser.add_argument('input_dir', help='Path to the input directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each file in the input directory
    for i, filename in enumerate(os.listdir(args.input_dir), start=1):
        if filename.lower().endswith(('.pdf', '.epub')):
            file_path = os.path.join(args.input_dir, filename)
            txt_path = os.path.join(args.output_dir, f'{i}.txt')
            process_file(file_path, txt_path)

if __name__ == '__main__':
    main()