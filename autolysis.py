import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet

from dotenv import load_dotenv
load_dotenv()

# Load the API token from the environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def load_data(file_path):
    """Load CSV data with encoding detection."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result.get('encoding', 'utf-8')  # Fallback to UTF-8
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {}
    }
    return analysis

def visualize_data(df):
    """Generate and save up to 3 visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    image_files = []
    
    for idx, column in enumerate(numeric_columns[:3]):  # Limit to 3 charts
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        image_file = f"{column}_distribution.png"
        plt.savefig(image_file)
        plt.close()
        image_files.append(image_file)
    
    return image_files

def generate_story_and_readme(analysis, image_files):
    """
    Generate a Markdown README.md with an automated narrative and chart references.
    """
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = (
        "Write a story-based analysis of the dataset based on the following summary. "
        "Integrate any patterns, trends, or anomalies into a creative narrative:\n"
        f"{analysis}"
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        narrative = response.json()['choices'][0]['message']['content']
    except Exception as e:
        narrative = f"Error generating story: {e}"

    # Write README.md
    with open("README.md", "w") as f:
        f.write("# Dataset Analysis\n\n")
        f.write("## Story\n")
        f.write(narrative + "\n\n")
        f.write("## Visualizations\n")
        for img in image_files:
            f.write(f"![{img}](./{img})\n")

def main(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # Load and analyze data
    df = load_data(file_path)
    if df.empty:
        print("Error: Dataset is empty. Exiting.")
        sys.exit(1)
    analysis = analyze_data(df)

    # Generate visualizations
    image_files = visualize_data(df)

    # Generate README with narrative and charts
    generate_story_and_readme(analysis, image_files)
    print("README.md and visualizations created successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
