import os

def combine_python_files(output_file: str):
    """Combine all Python files in the current workspace into a single text file."""
    with open(output_file, 'w') as outfile:
        for dirpath, _, filenames in os.walk('.'):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, 'r') as infile:
                        outfile.write(f"# Contents of {file_path}\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

if __name__ == "__main__":
    combine_python_files('combined_python_files.txt')
