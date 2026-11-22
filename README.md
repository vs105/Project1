# Project1

# Lok Sabha Bills Chatbot

An AI-powered chatbot that allows users to search, explore, and understand Lok Sabha bills using natural language.

The project uses **Python, Flask, Groq LLM, and a document retrieval pipeline** to answer questions based on legislative documents.

## 1. Project Overview

The Lok Sabha Bills Chatbot helps users understand Indian parliamentary legislation without manually reading lengthy PDF documents.

Users can ask questions about bills, retrieve relevant legislative information, and generate summaries of specific bills.

The project is designed to provide responses grounded in the available Lok Sabha bill documents.

## 2. Key Features

- **Natural Language Question Answering:** Ask questions about Lok Sabha bills in simple English.
- **Bill Search:** Find bills using their titles, years, or subjects.
- **Bill Summarization:** Generate summaries covering important provisions and clauses.
- **Clause-Based Retrieval:** Retrieve relevant sections and clauses from legislative documents.
- **Year-Based Filtering:** Search for bills introduced in a particular year.
- **Status-Based Queries:** Explore bills based on available legislative status information.
- **AI-Generated Responses:** Use the Groq API to generate readable answers from retrieved documents.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend | Flask |
| AI model integration | Groq API |
| Frontend | JavaScript / Vite |
| Data source | Lok Sabha bill PDFs |
| Document processing | PDF text extraction and cleaning |
| Information retrieval | Local preprocessed bill index |
| Version control | Git and GitHub |

## 4. How the Chatbot Works

The chatbot follows a document retrieval and AI response generation workflow.

**Step 1 — Data Collection**

Lok Sabha bill PDFs are collected and stored locally.

**Step 2 — Data Cleaning**

The extracted text is cleaned to remove unnecessary formatting and prepare it for processing.

**Step 3 — Document Chunking**

Long legislative documents are divided into smaller text chunks, including narrative content and operative clauses.

**Step 4 — Building the Knowledge Base**

The processed chunks are indexed to enable efficient searching and retrieval.

**Step 5 — User Query**

The user enters a question through the chatbot interface.

**Step 6 — Relevant Document Retrieval**

The system identifies the most relevant bills, clauses, and text chunks.

**Step 7 — AI Response Generation**

Relevant retrieved content is provided to the Groq-powered language model, which generates an answer.

**Step 8 — Displaying Results**

The chatbot displays the generated response to the user.

### Workflow

```text
Lok Sabha Bill PDFs
        |
        v
Text Extraction
        |
        v
Cleaning and Chunking
        |
        v
Local Knowledge Base
        |
        v
User Question
        |
        v
Bill and Clause Retrieval
        |
        v
Groq LLM
        |
        v
Generated Answer
        |
        v
Chatbot Interface
```

## 5. Project Repository

GitHub Repository:

https://github.com/vs105/Project1

## 6. Installation and Setup

### Prerequisites

Install the following software:

- Python 3
- Git
- Node.js and npm (for the Vite frontend)
- Visual Studio Code (recommended)
- A Groq API key

### Step 1 — Clone the Repository

Open PowerShell or the VS Code terminal.

```bash
git clone https://github.com/vs105/Project1.git
cd Project1
code .
```

If you already have the project, open its directory and run:

```bash
git pull origin main
```

### Step 2 — Create a Python Virtual Environment

In the project directory:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Python Dependencies

If `requirements.txt` is included in the repository:

```powershell
pip install -r requirements.txt
```

### Step 4 — Configure the Groq API Key

Each developer should use their own Groq API key.

Get your API key from:

https://console.groq.com/keys

If `.env.example` exists, copy it:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
```

The application must load this environment variable before making Groq API requests.

**Security:** Never upload `.env`, actual API keys, or credentials to GitHub. Keep `.env` in `.gitignore`.

### Step 5 — Prepare the Bill Knowledge Base

The chatbot requires its processed bill data and local search index.

If the repository already contains the necessary preprocessed data, no additional data preparation may be needed.

Otherwise, obtain the project's prepared knowledge base from your teammate or run the data collection, cleaning, and indexing scripts in the required order.

**Important:** The application cannot answer document-grounded questions correctly without the required bill data and index.

## 7. Running the Chatbot

### Terminal 1 — Start the Flask Backend

Activate the virtual environment and run:

```powershell
python server.py --port 8001
```

Expected backend address:

```text
http://127.0.0.1:8001
```

Keep this terminal running.

### Terminal 2 — Start the Frontend

Open another VS Code terminal.

Navigate to the directory containing the frontend's `package.json` file.

Install frontend dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

Vite will display a local URL, commonly:

```text
http://localhost:5173
```

If that port is already occupied, Vite may use another port such as `5174`.

Open the URL shown in your terminal.

## 8. Example Questions

You can test the chatbot using questions such as:

**Bill Search**

```text
List the bills introduced in 2024.
```

**Subject-Based Search**

```text
Which education-related bills were introduced in 2024?
```

**Bill Summarization**

```text
Summarize the Finance Bill 2024, covering its major provisions from different chapters.
```

**Clause-Based Questions**

```text
Explain the important clauses of the Finance Bill 2024.
```

**General Legislative Questions**

```text
What are the main objectives of the Juvenile Justice Bill 2015?
```

Results depend on the documents and metadata available in the local knowledge base.

## 9. Project Components

| File or component | Purpose |
|---|---|
| `server.py` | Flask backend and API server |
| `step4_talk.py` | Main chatbot query processing and retrieval logic |
| `.env.example` | Template for configuring environment variables |
| `.gitignore` | Prevents local and sensitive files from being committed |
| `requirements.txt` | Python dependency installation, if provided |
| `package.json` | Frontend dependencies and development scripts |
| Local bill index | Stores processed legislative content for retrieval |

## 10. Troubleshooting

### Groq API Key Not Found

Check that:

- `.env` contains `GROQ_API_KEY`.
- The key is valid.
- The application loads the environment variable.
- The backend has been restarted after changing the configuration.

### Knowledge Base Not Found

Ensure that the processed bill index exists in the location expected by the backend.

If it is missing, transfer the prepared data or rebuild the index using the project's processing scripts.

### Backend Connection Error

Check that Flask is running on port `8001` and that the frontend is configured to communicate with the correct backend address.

### npm Command Not Found

Install Node.js and reopen VS Code or your terminal.

### Frontend Port Already in Use

Use the alternative localhost URL displayed by Vite.

### Git Pull Fails

Check for uncommitted changes:

```bash
git status
```

Resolve any conflicts before continuing. Avoid force-pushing shared branches.

## 11. Team Collaboration

To get the latest version of the project:

```bash
git pull origin main
```

Before making changes, check the current branch and working tree:

```bash
git status
```

After completing your changes:

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

Always confirm that no API keys, `.env` files, large private datasets, or unnecessary generated files are being committed.

Team members should coordinate before editing the same files.

## 12. Current Project Status

The chatbot has a working document-processing and retrieval pipeline, with support for searching bills, retrieving clauses, and generating AI-powered responses.

The project has processed approximately:

- **5,932 source PDF documents**
- **3,678 indexed bills**
- **77,185 indexed text chunks**

These figures reflect the previously completed local processing run and may differ if the dataset is rebuilt.

Recent improvements include more targeted whole-bill retrieval and better handling of bill summaries across multiple clauses.

## 13. Future Improvements

- Improve coverage of all clauses in long-bill summaries.
- Add more advanced search and filtering.
- Improve citation and source-reference presentation.
- Optimize retrieval performance.
- Improve chatbot interface usability.
- Simplify setup and deployment for new team members.

## 14. Disclaimer

This chatbot is intended for educational and informational use.

AI-generated answers may contain inaccuracies or omit legislative details. For authoritative information, consult the original bill documents and official parliamentary sources.

---

**Project:** Lok Sabha Bills Chatbot  
**Repository:** https://github.com/vs105/Project1  
**Purpose:** AI-assisted exploration and understanding of Lok Sabha legislative documents.
