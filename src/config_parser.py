import configparser


def load_config_from_ini(file_path: str) -> dict | None:
    """
    Loads configuration from an INI file.

    Args:
        file_path: The path to the INI configuration file.

    Returns:
        A dictionary containing the configuration, or None if the file cannot be read.
    """
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        git_config = {}
        if "GitConfig" in config:
            repo_urls_str = config["GitConfig"].get("repo_urls", "")
            git_config["repo_urls"] = [
                url.strip() for url in repo_urls_str.split(",") if url.strip()
            ]
            git_config["company_identifier"] = (
                config["GitConfig"].get("company_identifier", "").strip()
            )
            git_config["months_back"] = config["GitConfig"].getint("months_back", None)
            git_config["deploy_dir"] = config["GitConfig"].get("deploy_dir", None)
        if "OpenAi" in config:
            git_config["openai_apikey"] = config["OpenAi"].get("openai_apikey", None)
        return git_config
    except Exception as e:
        print(f"Error reading INI file {file_path}: {e}")
        return None
