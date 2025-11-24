# 🔬 AI Research Assistant

An AI-powered toolkit that helps researchers search, analyze, summarize,
and generate insights from research papers---especially arXiv papers. It
automates the tedious parts of literature review so you can focus on
actual research.

## 🚀 Features

-   🔍 **Search & fetch arXiv papers** using simple queries
-   📄 **Download & read PDFs** automatically
-   🧠 **AI-powered summarization**, key-point extraction & section-wise
    analysis
-   📝 **Generate reports or slide-ready content**
-   🌐 Optional **web interface** (`app.py`) for interactive use
-   ⚙️ Modular code structure for extending your own research workflows

## 📦 Project Structure

    ai-research-assistant/
    ├── ai_researcher.py        # Main workflow
    ├── ai_researcher2.py       # Alternate workflow
    ├── arxiv_tool.py           # arXiv search & download tool
    ├── read_pdf.py             # PDF text extraction
    ├── write_pdf.py            # Generates PDF summaries/reports
    ├── app.py                  # Web interface
    ├── requirements.txt        # Dependencies
    ├── pyproject.toml          # Build config
    ├── uv.lock                 # Lock file
    └── .env                    # Environment variables (ignored by Git)

## 🛠️ Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/codewithkaran-21/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create a virtual environment (recommended)

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

If your workflow requires API keys or custom paths:

    cp .env.example .env   # if available

Fill the `.env` with relevant values.

## ▶️ Usage

### 🔎 Search for papers on arXiv

``` bash
python arxiv_tool.py --query "large language models"
```

### 📚 Read & analyze a local PDF

``` bash
python read_pdf.py --file path/to/paper.pdf
```

### 🧠 Generate a PDF report or slides

``` bash
python write_pdf.py --input analysis.json --output summary.pdf
```

### 🌐 Launch the web app

``` bash
python app.py
```

Open the browser at **http://localhost:5000**

## 📘 Example Use Cases

-   Summarize 10 latest papers on "GNNs" for a quick literature review
-   Convert a research paper into a slide deck
-   Extract abstract + methodology + results instantly from any PDF
-   Maintain a personal research directory with summaries

## 🤝 Contributing

Contributions are welcome!

1.  Fork the repo
2.  Create a feature branch
3.  Commit your changes
4.  Push & open a Pull Request

## 📄 License

MIT License

## 🙌 Acknowledgements

-   Built by **Karan Singh (codewithkaran-21)**
-   Thanks to **arXiv API** for open access
