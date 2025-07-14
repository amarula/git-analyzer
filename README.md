# Git Commit Article Generator (Standalone Python)

This Python script is a command-line tool designed to help you generate a blog article summarizing recent Git commits from one or more repositories, specifically filtering for commits made by your company. It supports both manual input and configuration via an INI file.

## Project Structure

```bash
git-analyzer/
├── amarula/                # Examples of configuration file from our organization
├── src/
│   ├── __init__.py         # Makes 'src' a Python package
│   ├── git_utils.py        # Contains functions for Git operations (clone, pull, analyze commits)
│   ├── article_generator.py # Contains logic for generating the article content
│   ├── config_parser.py    # Handles loading configuration from INI files
├── main.py                 # The main entry point of the application
├── pyproject.toml          # Project metadata and dependencies (modern Python packaging)
├── requirements.txt        # List of direct Python dependencies
├── README.md               # This file
├── .gitignore              # Specifies intentionally untracked files to ignore
```

## Dependencies

This project relies on the following key libraries:

* gitpython

Use requirements.txt to install it if needed.

```
pip install -r requirements.txt

```

## Features

* **Real Git Operations**: Clones specified Git repositories and analyzes their commit history.

* **Company-Specific Filtering**: Identifies commits made by a specified company identifier (e.g., email domain or author name).

* **Time-Based Analysis**: Filters commits within a configurable number of past months.

* **Multi-Repository Support**: Processes multiple Git repositories in a single run.

* **Article Generation**: Generates a summary article based on the analyzed commits.

* **INI File Configuration (Optional)**: Allows you to configure repository URLs, company identifier, and months to analyze using an INI file, making repeated runs easier.

* **Output to Console & File**: Prints the generated article to the console and offers to save it to a Markdown file.

## Prerequisites

Before running this tool, ensure you have the following installed on your system:

* **Python 3.x**: The script is written in Python 3. You can download it from [python.org](https://www.python.org/downloads/).

* **Git**: The command-line Git client must be installed and accessible in your system's PATH. You can download Git from [git-scm.com](https://git-scm.com/downloads).

## How to Use

### 1. Save the Script

Save the provided Python code as a `.py` file (e.g., `git_analyzer.py`).

### 2. (Optional) Create an INI Configuration File

For easier repeated use, you can create a configuration file named `config.ini` (or any other name you prefer) in the same directory as your Python script.

**Example `config.ini`:**

```ini
[GitConfig]
repo_urls = [https://github.com/torvalds/linux.git,https://github.com/kubernetes/kubernetes.git](https://github.com/torvalds/linux.git,https://github.com/kubernetes/kubernetes.git)
company_identifier = @linux.com # Or a name like 'Linus Torvalds'
months_back = 12
```

### 3. Execute the tool

The easy way to run it is to configure in a ini file and then run the tool.

```python
python main.py -f amarula/config.ini -s report.md
```

## TODO

- Use openai rest api to create a more consistent description of the work done in each project.
- Group the changes by Author and by Topic using for the topic the AI and commit messages information.
- Create a tool to post on social media like linkedin, X and facebook
- Add all amarula opensource projects
