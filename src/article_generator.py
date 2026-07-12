"""Article generator – builds Markdown blog posts from structured commit data.

Groups commits by repository and author, optionally enriching each author
section with an AI-generated summary of their contributions.
"""

import datetime
import textwrap
from collections import defaultdict
from datetime import datetime

import markdown_strings as md

from src.ai_utils import summarize_commit_messages
from src.git_utils import generate_commit_hyperlink


def _build_article_header(months_back: int, month_label: str | None,
                         report_date) -> str:
    """Return the markdown header for the article."""
    if month_label and report_date:
        report_date_str = report_date.strftime(
            '%Y-%m-%d') if hasattr(report_date, 'strftime') else str(report_date)
        return f"""
# Today {report_date_str} Developments: A Look at Our Codebase ({month_label} Monthly Review)

We're excited to share a summary of the significant progress made across our repositories
in the month of {month_label}. Our dedicated team has been busy pushing new features,
refining existing functionalities, and enhancing the overall stability of our products.

Here's a breakdown of key contributions by repository:

"""
    today = datetime.now()
    return f"""
# Today {today.strftime('%Y-%m-%d')} Developments: A Look at Our Codebase ({months_back} Months Review)

We're excited to share a summary of the significant progress made across our repositories
in the last {months_back} months. Our dedicated team has been busy pushing new features,
refining existing functionalities, and enhancing the overall stability of our products.

Here's a breakdown of key contributions by repository:

"""


def _format_commit_line(commit: dict, repo_url: str) -> str:
    """Format a single commit as a markdown bullet with hyperlink."""
    first_line = (commit['message'].split('\n')[0]
                  if commit['message'] else '(No message)')
    first_line = first_line.replace('_', r'\_')
    hyperlink = generate_commit_hyperlink(repo_url, commit['sha1'])
    return (f"- **{commit['author_name']}** on "
            f"{commit['date'].split('T')[0]}: [{first_line}]({hyperlink})\n")


def _build_author_section(author_name: str, author_commits: list,
                          repo_url: str, ai_config: dict | None,
                          months_back: int) -> str:
    """Build the markdown section for a single author's contributions."""
    section = ''

    if ai_config:
        all_messages = '\n'.join([
            c['message'] if c['message'] else '(No message provided)'
            for c in author_commits
        ])
        ai_summary = summarize_commit_messages(
            ai_config['key'], all_messages, months_back,
            author_name, ai_config['model'],
        )
        ai_summary = md.esc_format(ai_summary, esc=True)
        if ai_summary:
            wrapped = textwrap.fill(ai_summary, width=100)
            section += f"\n**{author_name}**: {wrapped}\n\n"

    section += f"#### Here the commits of **{author_name}** in detail:\n\n"
    data = sorted(author_commits, key=lambda x: (x['author_name'], x['date']))
    for commit in data:
        section += _format_commit_line(commit, repo_url)
    section += '\n'
    return section


def generate_article_content(
    commit_data: list[dict],
    months_back: int,
    ai_config: dict | None = None,
    month_label: str = None,
    report_date=None,
) -> str:
    """Generates a blog article based on commit data.

    Args:
        commit_data: Structured commit data.
        months_back: The number of months the analysis covered.
        ai_config: Optional dict with 'key' (OpenAI API key) and 'model'
                   (OpenAI model name) for commit summarization.
        month_label: Optional label for a single-month report (e.g. "2026-06").
        report_date: Optional datetime to use as the report date (end of month).

    Returns:
        A string containing the blog article.

    """
    if not commit_data:
        return 'No relevant commits found to generate an article.'

    article_content = _build_article_header(months_back, month_label, report_date)

    for repo in commit_data:
        if 'error' in repo or not repo['commits']:
            continue

        article_content += f"## {repo['repo_name']}\n\n"
        article_content += f"Repository URL: {repo['repo_url']}\n\n"

        commits_by_author: dict = defaultdict(list)
        for commit in repo['commits']:
            commits_by_author[commit['author_name']].append(commit)

        if not commits_by_author:
            continue

        article_content += '### Summary of the contributions by author:\n\n'

        for author_name, author_commits in sorted(commits_by_author.items()):
            article_content += _build_author_section(
                author_name, author_commits, repo['repo_url'],
                ai_config, months_back,
            )

    article_content += """
This overview highlights the continuous effort and innovation from our development team. We look forward to bringing even more exciting updates in the future!

---
*Generated by the Git Commit Article Generator*
"""
    return article_content
