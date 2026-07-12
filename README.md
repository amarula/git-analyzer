# Git Commit Article Generator

A command-line tool that clones Git repositories, analyzes commit history filtered by
company identifier, and generates a blog-style summary article — optionally enriched
with AI-generated author summaries.

## Project Structure

```
git-analyzer/
├── amarula/                  # Example INI configuration files
├── src/
│   ├── __init__.py           # Makes 'src' a Python package
│   ├── git_utils.py          # Git operations (clone, pull, analyze commits)
│   ├── article_generator.py  # Blog article content generation
│   ├── config_parser.py      # INI configuration file loader
│   ├── ai_utils.py           # OpenAI commit-message summarization
├── main.py                   # Entry point
├── pyproject.toml            # Project metadata
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## Prerequisites

- **Python 3.x**
- **Git** (must be in your `PATH`)

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|---|---|
| `GitPython` | Clone repos, iterate commit history |
| `markdown_strings` | Markdown escaping in generated articles |
| `langchain-openai` | AI-powered commit message summarization (optional) |

## Features

- **Real Git Operations** — clones repositories and analyzes their commit history.
- **Company-Specific Filtering** — matches commits by author name, email domain,
  or `Signed-off-by` trailer.
- **Per-Month Reports** — when a multi-month range is requested, each calendar
  month gets its own report with the date set to the last day of that month.
- **Multi-Repository Support** — processes any number of repositories in a single run.
- **AI Summaries (optional)** — summarize each author's contributions via OpenAI.
- **INI File Configuration** — drive repeated runs from a config file.
- **Output to Console & File** — prints the article and optionally saves it as
  Markdown.

## Command-Line Reference

```
python main.py [OPTIONS]
```

| Flag | Argument | Description |
|---|---|---|
| `-r`, `--repo-urls` | `URL,URL,…` | Comma-separated list of Git repository URLs. |
| `-c`, `--company-identifier` | `STRING` | String to identify company commits (e.g. `@mycompany.com`). |
| `-m`, `--months-back` | `INT` | Number of months to look back. |
| `-f`, `--config-file` | `PATH` | Path to an INI configuration file. Overrides `-r`, `-c`, `-m`. |
| `-s`, `--save-to-file` | `FILE` | Save article(s) to file. Defaults to `git_report.md` if no filename is given. |
| `-d`, `--deploy-dir` | `DIR` | Directory to clone repositories into (default: `deploy`). |
| `-k`, `--ai-key` | `KEY` | OpenAI API key for per-author commit summaries. |

If `-r`, `-c`, or `-m` are omitted and no config file is used, the tool prompts
interactively.

## Per-Month Report Behaviour

When you pass `-m N` with **N > 1**, the tool generates **N separate reports** —
one per complete calendar month. Each report is dated with the last day of that month
and covers commits from the 1st to the last day (inclusive).

For example, running on **July 12, 2026** with `-m 3` produces:

| Report | Date Range | Title Date |
|---|---|---|
| 2026-04 | 2026-04-01 → 2026-04-30 | **2026-04-30** |
| 2026-05 | 2026-05-01 → 2026-05-31 | **2026-05-31** |
| 2026-06 | 2026-06-01 → 2026-06-30 | **2026-06-30** |

When you pass `-m 1`, the legacy behaviour is preserved: a single report covering
roughly the last 30 days, dated with today's date.

When saving with `-s` in multi-month mode, files are named automatically:

```
-s report.md   →   report_2026-04.md, report_2026-05.md, report_2026-06.md
```

## INI Configuration File

Create a `.ini` file to avoid typing arguments repeatedly:

```ini
[GitConfig]
repo_urls = https://github.com/torvalds/linux.git,https://github.com/u-boot/u-boot.git
company_identifier = @mycompany.com
months_back = 3
deploy_dir = deploy

[OpenAi]
ai_apikey = sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ai_model = gpt-4o
```

- `[GitConfig]` — required section with `repo_urls`, `company_identifier`,
  `months_back`, and optionally `deploy_dir`.
- `[OpenAi]` — optional section. When present, each author's commits are
  summarized via the OpenAI API. `ai_model` defaults to `gpt-3.5-turbo` if omitted.

## Usage Examples

### Quick interactive run

```bash
python main.py
# The tool will prompt for repo URLs, company identifier, and months.
```

### One month, explicit arguments

```bash
python main.py -r https://github.com/torvalds/linux.git -c @mycompany.com -m 1
```

### Three months, save per-month reports

```bash
python main.py -r https://github.com/torvalds/linux.git -c @mycompany.com -m 3 -s
# Produces: git_report_2026-04.md, git_report_2026-05.md, git_report_2026-06.md
```

### Three months, save to a custom base name

```bash
python main.py -r https://github.com/torvalds/linux.git -c @mycompany.com -m 3 -s linux.md
# Produces: linux_2026-04.md, linux_2026-05.md, linux_2026-06.md
```

### Drive everything from a config file

```bash
python main.py -f amarula/config.ini -s report.md
```

### With AI-powered author summaries

```bash
python main.py -f amarula/config.ini -s report.md -k sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## License

MIT — see [LICENSE](LICENSE).
