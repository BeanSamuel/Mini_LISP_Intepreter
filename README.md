# Mini_LISP_Intepreter

## Introduction
This Python script processes `.lsp` files, allowing users to execute specified files or process all `.lsp` files in a designated directory. The tool is designed to parse and evaluate expressions from Lisp files.

## Features
- **Custom File Execution**: Specify `.lsp` files to process via command-line arguments.
- **Default Directory Processing**: Automatically processes all `.lsp` files in a specified directory if no files are provided.
- **Error Handling**: Validates file existence and provides error messages to prevent crashes.

## Usage
### Run the Script
Ensure Python 3 is installed and save the script as `main.py`.

### Specify Files
Run the following command to process one or more `.lisp` files:
```bash
python3 main.py file1.lisp file2.lisp
```

### Default Directory
If no files are specified, the script processes all `.lisp` files in the `public_test_data` directory:
```bash
python3 main.py
```

## File Structure
- `main.py`: The main script for reading files, parsing content, and execution.
- `public_test_data`: Directory containing public test `.lisp` files.

## Developer Notes
### Key Classes and Functions
1. `EnvironmentStack`
   - Simulates the execution environment stack.
   - Extend this class as needed for additional functionality.

2. `parse_parenthesized_expression(content)`
   - Parses parenthesized expressions in the content.
   - Replace with the actual implementation.

### Dependencies
- No external dependencies. The script runs with Python's standard library.

## Example Output
When processing files, the output will include the following format:
```
==================== Processing File: file1.lisp ====================
<Parsed and evaluated results>
```
If an error occurs, an error message will be displayed.


