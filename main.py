import argparse

# Import functions from the new modules
from src.git_utils import analyze_real_git_commits
from src.article_generator import generate_article_content
from src.config_parser import load_config_from_ini


def main():
    """
    Main function to run the Git commit analysis and article generation tool.
    Supports optional configuration from an INI file and command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a blog article summarizing Git commits from repositories."
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
        "--openai-key",
        help="Pass the openai key for create commit summary by Author and repo.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    repo_urls = []
    company_identifier = ""
    months_back = None
    save_file_name = None
    deploy_dir = None
    openai_key = None

    config_loaded_successfully = False

    # Attempt to load from INI file if specified
    if args.config_file:
        config_data = load_config_from_ini(args.config_file)
        if config_data:
            repo_urls = config_data.get("repo_urls", [])
            company_identifier = config_data.get("company_identifier", "")
            months_back = config_data.get("months_back", None)
            deploy_dir = config_data.get("deploy_dir", None)
            openai_apikey = config_data.get("openai_apikey", None)

            config_loaded_successfully = True
            print(f"Configuration loaded from {args.config_file}.")
        else:
            print(
                f"Warning: Failed to load configuration from {args.config_file}."
                " Proceeding with command-line arguments or prompts."
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
        if args.openai_key:
            openai_key = args.openai_key

    if args.deploy_dir:
        deploy_dir = args.deploy_dir

    if deploy_dir is None:
        deploy_dir = "deploy"

    if args.save_to_file is not None:
        save_file_name = args.save_to_file

    if not repo_urls:
        repo_urls_input = input(
            "Enter Git repository URLs (comma-separated, e.g.,"
            "https://github.com/org/repo1.git,https://github.com/org/repo2.git): "
        ).strip()
        repo_urls = [url.strip() for url in repo_urls_input.split(",") if url.strip()]

    if not repo_urls:
        print("No repository URLs provided. Exiting.")
        return

    if not company_identifier:
        company_identifier = input(
            "Enter your company identifier (e.g., @mycompany.com or 'My Company Name'): "
        ).strip()

    if not company_identifier:
        print("Company identifier cannot be empty. Exiting.")
        return

    if months_back is None:
        while True:
            try:
                months_back = int(
                    input("Enter number of months back to analyze (e.g., 3): ").strip()
                )
                if months_back <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for months.")

    print("\nStarting real Git analysis...")
    analysis_result = analyze_real_git_commits(
        repo_urls, company_identifier, months_back, deploy_dir
    )

    if "error" in analysis_result:
        print(f"\nError during Git analysis: {analysis_result['error']}")
        return

    commit_data = analysis_result.get("commit_data", [])

    article = generate_article_content(commit_data, months_back, openai_key)

    print("\n--- Generated Article ---")
    print(article)
    print("\n--- End of Article ---")

    if save_file_name:
        try:
            with open(save_file_name, "w", encoding="utf-8") as f:
                f.write(article)
            print(f"Article automatically saved to {save_file_name}")
        except Exception as e:
            print(f"Error automatically saving file {save_file_name}: {e}")
    else:
        save_option = (
            input("\nDo you want to save the article to a file? (yes/no): ")
            .lower()
            .strip()
        )
        if save_option == "yes":
            file_name = input("Enter desired filename (e.g., git_report.md): ").strip()
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
