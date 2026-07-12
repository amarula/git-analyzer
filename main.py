"""Git Commit Analyzer – Generate blog articles from repository commit history.

This tool clones Git repositories, extracts commits matching a company
identifier, optionally summarises them via an AI backend, and produces a
Markdown article suitable for publishing as an engineering blog post.
"""

import argparse
from datetime import datetime, timedelta

from src.article_generator import generate_article_content
from src.config_parser import load_config_from_ini

# Import functions from the new modules
from src.git_utils import analyze_real_git_commits


def get_calendar_months(
        months_back: int) -> list[tuple[datetime, datetime, str]]:
    """Return a list of (since_date, until_date, month_label) for the last N
    complete calendar months.

    Args:
        months_back: Number of complete calendar months to look back.

    Returns:
        A list of tuples (since_date, until_date, month_label) where each entry
        covers exactly one calendar month, from oldest to newest.

    """
    today = datetime.now()
    months = []
    for i in range(months_back, 0, -1):
        year = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year -= 1
        # First day of the target month
        since = datetime(year, month, 1)
        # Last day of the target month
        if month == 12:
            until = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            until = datetime(year, month + 1, 1) - timedelta(days=1)
        until = until.replace(hour=23, minute=59, second=59)
        label = f"{year}-{month:02d}"
        months.append((since, until, label))
    return months


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the configured argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate a blog article summarizing Git commits from repositories.',
    )
    parser.add_argument(
        '-r', '--repo-urls',
        help='Comma-separated list of Git repository URLs.',
        type=str,
    )
    parser.add_argument(
        '-c', '--company-identifier',
        help='String to identify company commits (e.g., email domain or "My Company Name").',
        type=str,
    )
    parser.add_argument(
        '-m', '--months-back',
        help='Number of months back to analyze commits.',
        type=int,
    )
    parser.add_argument(
        '-f', '--config-file',
        help='Path to an INI configuration file. If provided and successfully loaded, '
             'it will override other core parameters (-r, -c, -m).',
        type=str,
    )
    parser.add_argument(
        '-s', '--save-to-file',
        help='Automatically save the generated article to a file (provide filename).',
        type=str,
        nargs='?',
        const='git_report.md',
    )
    parser.add_argument(
        '-d', '--deploy-dir',
        help="Name of the directory to clone repositories into (default: 'deploy').",
        type=str,
        default='deploy',
    )
    parser.add_argument(
        '-k', '--ai-key',
        help='Pass the ai key for create commit summary by Author and repo.',
        type=str,
        default=None,
    )
    return parser


def _apply_ini_config(config: dict, config_data: dict) -> None:
    """Populate *config* from a successfully loaded INI *config_data*."""
    config['repo_urls'] = config_data.get('repo_urls', [])
    config['company_identifier'] = config_data.get('company_identifier', '')
    config['months_back'] = config_data.get('months_back', None)
    config['deploy_dir'] = config_data.get('deploy_dir', None)
    config['ai_key'] = config_data.get('ai_apikey', None)
    config['ai_model'] = config_data.get('ai_model', None)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Overlay CLI arguments onto *config* where they are not None."""
    if args.repo_urls:
        config['repo_urls'] = [
            url.strip() for url in args.repo_urls.split(',') if url.strip()
        ]
    if args.company_identifier:
        config['company_identifier'] = args.company_identifier.strip()
    if args.months_back is not None:
        config['months_back'] = args.months_back
    if args.deploy_dir:
        config['deploy_dir'] = args.deploy_dir
    if args.ai_key:
        config['ai_key'] = args.ai_key


def _prompt_for_missing(config: dict) -> bool:
    """Prompt interactively for missing required values.

    Returns True if all required values are present, False if the user
    aborted (and the caller should exit).
    """
    if not config['repo_urls']:
        raw = input(
            'Enter Git repository URLs (comma-separated, e.g.,'
            'https://github.com/org/repo1.git,https://github.com/org/repo2.git): ',
        ).strip()
        config['repo_urls'] = [
            url.strip() for url in raw.split(',') if url.strip()
        ]

    if not config['repo_urls']:
        print('No repository URLs provided. Exiting.')
        return False

    if not config['company_identifier']:
        config['company_identifier'] = input(
            "Enter your company identifier (e.g., @mycompany.com or 'My Company Name'): ",
        ).strip()

    if not config['company_identifier']:
        print('Company identifier cannot be empty. Exiting.')
        return False

    if config['months_back'] is None:
        while True:
            try:
                config['months_back'] = int(
                    input('Enter number of months back to analyze (e.g., 3): ').strip(),
                )
                if config['months_back'] <= 0:
                    raise ValueError
                break
            except ValueError:
                print('Invalid input. Please enter a positive integer for months.')

    return True


def _resolve_configuration(args: argparse.Namespace) -> dict | None:
    """Build the final configuration dict from INI file, CLI args and prompts.

    Returns the config dict, or None when the user aborts during interactive
    prompts (missing required values).
    """
    config: dict = {
        'repo_urls': [],
        'company_identifier': '',
        'months_back': None,
        'deploy_dir': 'deploy',
        'ai_key': None,
        'ai_model': None,
        'save_file_name': None,
    }

    config_loaded = False

    if args.config_file:
        ini_data = load_config_from_ini(args.config_file)
        if ini_data:
            _apply_ini_config(config, ini_data)
            config_loaded = True
            print(f"Configuration loaded from {args.config_file}.")
        else:
            print(
                f"Warning: Failed to load configuration from {args.config_file}."
                ' Proceeding with command-line arguments or prompts.',
            )

    if not config_loaded:
        _apply_cli_overrides(config, args)

    # deploy_dir and save_to_file always honour the CLI arg last
    if args.deploy_dir:
        config['deploy_dir'] = args.deploy_dir
    if args.save_to_file is not None:
        config['save_file_name'] = args.save_to_file

    if not _prompt_for_missing(config):
        return None

    if config['ai_key'] and not config['ai_model']:
        config['ai_model'] = 'gpt-3.5-turbo'

    return config


def _save_article(article: str, save_file_name: str,
                  month_label: str | None) -> None:
    """Save *article* to disk, deriving per-month filenames when needed."""
    if save_file_name:
        if month_label is not None:
            base, ext = (save_file_name.rsplit('.', 1)
                         if '.' in save_file_name
                         else (save_file_name, 'md'))
            month_file = f"{base}_{month_label}.{ext}"
        else:
            month_file = save_file_name
        try:
            with open(month_file, 'w', encoding='utf-8') as f:
                f.write(article)
            print(f"Article automatically saved to {month_file}")
        except OSError as e:
            print(f"Error automatically saving file {month_file}: {e}")
        return

    save_option = (
        input('\nDo you want to save the article to a file? (yes/no): ')
        .lower()
        .strip()
    )
    if save_option != 'yes':
        return

    file_name = input(
        'Enter desired filename (e.g., git_report.md): '
    ).strip()
    if not file_name:
        file_name = 'git_report.md'
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(article)
        print(f"Article saved to {file_name}")
    except OSError as e:
        print(f"Error saving file: {e}")


def _process_month(since_date, until_date, month_label,
                   config: dict) -> str | None:
    """Analyse commits and generate an article for a single month.

    Returns the generated article string, or None on error.
    """
    print('\nStarting real Git analysis...')
    analysis_result = analyze_real_git_commits(
        config['repo_urls'],
        config['company_identifier'],
        config['months_back'],
        config['deploy_dir'],
        date_range=(since_date, until_date),
    )

    if 'error' in analysis_result:
        print(f"\nError during Git analysis: {analysis_result['error']}")
        return None

    commit_data = analysis_result.get('commit_data', [])
    ai_cfg = (
        {'key': config['ai_key'], 'model': config['ai_model']}
        if config['ai_key']
        else None
    )
    article = generate_article_content(
        commit_data,
        config['months_back'],
        ai_config=ai_cfg,
        month_label=month_label,
        report_date=until_date,
    )

    print('\n--- Generated Article ---')
    print(article)
    print('\n--- End of Article ---')
    return article


def main():
    """Entry point for the Git commit analysis and article generation tool."""
    args = _build_arg_parser().parse_args()

    config = _resolve_configuration(args)
    if config is None:
        return

    calendar_months = (
        get_calendar_months(config['months_back'])
        if config['months_back'] > 1
        else [(None, None, None)]
    )

    for since_date, until_date, month_label in calendar_months:
        if month_label is not None:
            print(f"\n--- Processing month: {month_label} "
                  f"({since_date.strftime('%Y-%m-%d')} to "
                  f"{until_date.strftime('%Y-%m-%d')}) ---")

        article = _process_month(since_date, until_date, month_label, config)
        if article is None:
            return

        _save_article(article, config['save_file_name'], month_label)


if __name__ == '__main__':
    main()
