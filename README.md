AI Research Assistant

A tool to streamline research paper exploration using AI and arXiv analysis

Overview

The AI Research Assistant is a Python-based utility designed to help researchers interactively explore, summarise, and manage academic papers (especially from Cornell University’s arXiv) by combining PDF reading, metadata extraction and AI-powered summarisation. It aims to reduce the time spent digging through large numbers of research articles, allowing you to focus on insights and next-steps.

Features

Fetch metadata and download PDFs from arXiv via the arxiv_tool.py module.

Read and parse PDF content (using read_pdf.py) to extract key sections.

Use AI to generate summaries, extract bullet-points, or create slide-ready content.

Generate slide decks or reports via write_pdf.py.

Web interface via app.py for interactive exploration (optional).

Modular codebase for extending to other sources or adding custom workflows.

Getting Started
Prerequisites

Python 3.8+

Internet connection for arXiv access and AI calls (if used).

An AI model/API key if you’re using a hosted AI-service (adjust code accordingly).

Installation

1. Clone the repository:

git clone https://github.com/codewithkaran-21/ai-research-assistant.git  
cd ai-research-assistant  

2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

If your workflow requires API keys or custom paths:

cp .env.example .env   # if available

Fill the .env with relevant values.
▶️ Usage
🔎 Search for papers on arXiv

python arxiv_tool.py --query "large language models"

📚 Read & analyze a local PDF

python read_pdf.py --file path/to/paper.pdf

🧠 Generate a PDF report or slides

python write_pdf.py --input analysis.json --output summary.pdf

🌐 Launch the web app

python app.py

Open the browser at:
http://localhost:5000
📘 Example Use Cases

    Summarize 10 latest papers on "GNNs" for a quick literature review

    Convert a research paper into a slide deck

    Extract abstract + methodology + results instantly from any PDF

    Maintain a personal research directory with summaries

