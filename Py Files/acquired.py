#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import datetime
import requests
import re
import pandas as pd
from env import token, username

repos = []  # Global variable for repository names
words = []
languages = []

def get_github_data(token, username):
    global repos  # Declare repos as a global variable

    def get_repository_names():
        global repos  # Declare repos as a global variable

        # GitHub API endpoint
        url = "https://api.github.com/search/repositories"

        # Parameters for the API request
        params = {
            "q": "stars:>0",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
        }

        # Authentication headers
        headers = {"Authorization": f"token {token}", "User-Agent": username}

        # Send a GET request to the GitHub API
        response = requests.get(url, params=params, headers=headers)

        # Check if the response is successful
        if response.status_code == 200:
            # Get the response data
            response_data = response.json()

            # Extract the repository names
            repos = [repo["full_name"] for repo in response_data["items"]]

            # Create a DataFrame with the repository names
            df = pd.DataFrame({"Repository": repos})

            return df
        else:
            print("Failed to retrieve repository names from the GitHub API")
            return pd.DataFrame()

    def get_repo_language(repo: str) -> str:
        global repos  # Access the global repos variable
        url = f"https://api.github.com/repos/{repo}"
        headers = {"Authorization": f"token {token}", "User-Agent": username}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            repo_info = response.json()
            if "language" in repo_info:
                language = repo_info["language"]
                repos.append(repo)  # Append the repository name to repos
                return language
        return ""

    def get_languages_for_repos(repos: list):
        for repo in repos:
            language = get_repo_language(repo)
            if language:
                print(f"Repository: {repo}, Language: {language}")
            else:
                print(f"Failed to retrieve language for repository: {repo}")

    def get_readme_text():
        # Get the repository names
        df_repos = get_repository_names()

        # Scrape README files and store their content in a dictionary
        readme_data = {'Repository': [], 'Readme': [], 'Language': []}
        for repo in df_repos['Repository']:
            # Get the URL of the README file
            url = f"https://api.github.com/repos/{repo}/readme"

            # Authentication headers
            headers = {"Authorization": f"token {token}", "User-Agent": username}

            # Send a GET request to the URL with authentication
            response = requests.get(url, headers=headers)

            # Check if the response is successful
            if response.status_code == 200:
                # Get the content from the response
                response_data = response.json()
                readme_content = response_data['content']

                                # Decode the base64 encoded content
                import base64
                readme_text = base64.b64decode(readme_content).decode('utf-8')

                # Store the repository name, readme, and language
                readme_data['Repository'].append(repo)
                readme_data['Readme'].append(readme_text)

                # Get the language for the repository
                language = get_repo_language(repo)
                readme_data['Language'].append(language)
            else:
                print(f"Failed to retrieve README for repository: {repo}")

        # Create a DataFrame from the scraped data
        df_readme = pd.DataFrame(readme_data)

        # Return the DataFrame
        return df_readme

    # Call the inner functions to retrieve the data
    df = get_readme_text()

    # Return the resulting DataFrame
    return df