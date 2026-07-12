import argparse
from datetime import datetime, timedelta

from src.article_generator import generate_article_content
from src.config_parser import load_config_from_ini

# Import functions from the new modules
from src.git_utils import analyze_real_git_commits


def get_calendar_months(months_back: int) -> list[tuple[datetime, datetime, str]]:
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


def main():
    """Main function to run the Git commit analysis and article generation tool.
    Supports optional configuration from an INI file and command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a blog article summarizing Git commits from repositories.",
    )
    parser.add_argument(
        "-r",
        "--repo-urls",
        help="Comma-separated list of Git repository URLs.",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--company-identifier",
        help='String to identify company commits (e.g., email domain or "My Company Name").',
        type=str,
    )
    parser.add_argument(
        "-m",
        "--months-back",
        help="Number of months back to analyze commits.",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--config-file",
        help="Path to an INI configuration file. If provided and successfully loaded, "
             "it will override other core parameters (-r, -c, -m).",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--save-to-file",
        help="Automatically save the generated article to a file (provide filename).",
        type=str,
        nargs="?",  # Allows the argument to be optional, if present without value, it's None
        const="git_report.md",  # Default value if -s is present without an argument
    )
    parser.add_argument(
        "-d",
        "--deploy-dir",
        help="Name of the directory to clone repositories into (default: 'deploy').",
        type=str,
        default="deploy",
    )
    parser.add_argument(
        "-k",
        "--ai-key",
        help="Pass the ai key for create commit summary by Author and repo.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    repo_urls = []
    company_identifier = ""
    months_back = None
    save_file_name = None
    deploy_dir = None
    ai_key = None

    config_loaded_successfully = False

    # Attempt to load from INI file if specified
    if args.config_file:
        config_data = load_config_from_ini(args.config_file)
        if config_data:
            repo_urls = config_data.get("repo_urls", [])
            company_identifier = config_data.get("company_identifier", "")
            months_back = config_data.get("months_back", None)
            deploy_dir = config_data.get("deploy_dir", None)
            ai_key = config_data.get("ai_apikey", None)
            ai_model = config_data.get("ai_model", None)

            config_loaded_successfully = True
            print(f"Configuration loaded from {args.config_file}.")
        else:
            print(
                f"Warning: Failed to load configuration from {args.config_file}."
                " Proceeding with command-line arguments or prompts.",
            )

    # If config file was NOT successfully loaded, or not provided, then use CLI args
    if not config_loaded_successfully:
        if args.repo_urls:
            repo_urls = [
                url.strip() for url in args.repo_urls.split(",") if url.strip()
            ]
        if args.company_identifier:
            company_identifier = args.company_identifier.strip()
        if args.months_back is not None:
            months_back = args.months_back
        if args.deploy_dir:
            deploy_dir = args.deploy_dir
        if args.ai_key:
            ai_key = args.ai_key
        if args.ai_model:
            ai_model = args.ai_model

    if args.deploy_dir:
        deploy_dir = args.deploy_dir

    if deploy_dir is None:
        deploy_dir = "deploy"

    if args.save_to_file is not None:
        save_file_name = args.save_to_file

    if not repo_urls:
        repo_urls_input = input(
            "Enter Git repository URLs (comma-separated, e.g.,"
            "https://github.com/org/repo1.git,https://github.com/org/repo2.git): ",
        ).strip()
        repo_urls = [url.strip() for url in repo_urls_input.split(",") if url.strip()]

    if not repo_urls:
        print("No repository URLs provided. Exiting.")
        return

    if not company_identifier:
        company_identifier = input(
            "Enter your company identifier (e.g., @mycompany.com or 'My Company Name'): ",
        ).strip()

    if not company_identifier:
        print("Company identifier cannot be empty. Exiting.")
        return

    if months_back is None:
        while True:
            try:
                months_back = int(
                    input("Enter number of months back to analyze (e.g., 3): ").strip(),
                )
                if months_back <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for months.")

    if ai_key and not ai_model:
        ai_model = "gpt-3.5-turbo"

    # Determine which months to process.
    # For a single month, use the original behaviour (single report with
    # today's date).  For multiple months, iterate over complete calendar
    # months so each report covers exactly one month and is dated with the
    # last day of that month.
    if months_back > 1:
        calendar_months = get_calendar_months(months_back)
    else:
        calendar_months = [(None, None, None)]  # sentinel: use legacy behaviour

    for since_date, until_date, month_label in calendar_months:
        if month_label is not None:
            print(f"\n--- Processing month: {month_label} "
                  f"({since_date.strftime('%Y-%m-%d')} to "
                  f"{until_date.strftime('%Y-%m-%d')}) ---")

        print("\nStarting real Git analysis...")
        analysis_result = analyze_real_git_commits(
            repo_urls,
            company_identifier,
            months_back,
            deploy_dir,
            since_date=since_date,
            until_date=until_date,
        )

        if "error" in analysis_result:
            print(f"\nError during Git analysis: {analysis_result['error']}")
            return

        commit_data = analysis_result.get("commit_data", [])

        article = generate_article_content(
            commit_data,
            months_back,
            ai_key,
            ai_model,
            month_label=month_label,
            report_date=until_date,
        )

        print("\n--- Generated Article ---")
        print(article)
        print("\n--- End of Article ---")

        if save_file_name:
            # When processing multiple months, derive per-month filenames.
            if month_label is not None:
                base, ext = (save_file_name.rsplit(".", 1)
                             if "." in save_file_name
                             else (save_file_name, "md"))
                month_file = f"{base}_{month_label}.{ext}"
            else:
                month_file = save_file_name
            try:
                with open(month_file, "w", encoding="utf-8") as f:
                    f.write(article)
                print(f"Article automatically saved to {month_file}")
            except Exception as e:
                print(f"Error automatically saving file {month_file}: {e}")
        else:
            save_option = (
                input("\nDo you want to save the article to a file? (yes/no): ")
                .lower()
                .strip()
            )
            if save_option == "yes":
                file_name = input(
                    "Enter desired filename (e.g., git_report.md): "
                ).strip()
                if not file_name:
                    file_name = "git_report.md"
                try:
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(article)
                    print(f"Article saved to {file_name}")
                except Exception as e:
                    print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
