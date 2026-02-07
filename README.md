# CodeFlow AI 🚀

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/AI-Powered-purple.svg" alt="AI Powered">
</p>

**CodeFlow AI** is an AI-powered educational platform that helps students and developers understand code execution through **animated visualizations** and **natural language explanations**.

## ✨ Features

### 🎬 Visual Code Execution
- **Step-by-step visualization** of code execution
- **Animated arrays** with pointer tracking
- **Variable state tracking** with real-time updates
- **Control flow diagrams** showing program logic
- **Timeline navigation** for easy step browsing

### 🤖 AI Tutor
- **Natural language explanations** for each step
- **"Why this happens"** contextual insights
- **Common mistakes** warnings
- **Ask questions** about any part of the code
- **Complexity analysis** (Time & Space)

### 📝 Code Analysis
- **Safe AST-based parsing** (no code execution)
- **Support for Python** algorithms and DSA
- **Pre-built examples** (Bubble Sort, Binary Search, etc.)
- **Custom input values** for testing

### 🎮 Interactive Controls
- **Play/Pause** animation
- **Step forward/backward**
- **Speed control** (0.5x to 3x)
- **Keyboard shortcuts**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- A modern web browser

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd codeflow_ai
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional):**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key (optional)
```

5. **Start the server:**
```bash
cd backend
python main.py
```

6. **Open in browser:**
```
http://localhost:8000
```

---

## 📁 Project Structure

```
codeflow_ai/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ast_parser.py      # Safe AST code parsing
│   │   ├── step_generator.py  # Execution step generation
│   │   └── ai_engine.py       # AI explanation engine
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   └── models.py              # Pydantic data models
├── frontend/
│   ├── index.html             # Main UI
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend logic
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend application |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/analyze` | Analyze code and generate steps |
| `POST` | `/api/explain/step/{id}` | Get explanation for specific step |
| `POST` | `/api/ask` | Ask AI a question about code |
| `POST` | `/api/complexity` | Get complexity analysis |
| `GET` | `/api/examples` | Get pre-built code examples |
| `GET` | `/api/settings` | Get current settings |
| `PUT` | `/api/settings` | Update AI style |

### Example API Usage

```python
import requests

# Analyze code
response = requests.post("http://localhost:8000/api/analyze", json={
    "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
    """,
    "input_values": {"arr": [64, 34, 25, 12]}
})

data = response.json()
print(f"Generated {len(data['steps'])} steps")
```

---

## 📊 Execution Step JSON Schema

Each execution step follows this structure:

```json
{
    "step_id": 1,
    "step_type": "assign",
    "line_number": 3,
    "code_snippet": "n = len(arr)",
    "description": "Assign len(arr) to n",
    "state_before": {},
    "state_after": {"n": 4, "arr": [64, 34, 25, 12]},
    "visualization": {
        "highlight_lines": [3],
        "highlight_variables": ["n"],
        "array_indices": [],
        "animation_type": "value_change"
    },
    "ai_context": {
        "action": "assignment",
        "targets": ["n"],
        "value": "len(arr)"
    }
}
```

---

## 🎓 Supported Algorithms

CodeFlow AI comes with pre-built examples for:

- **Sorting:** Bubble Sort, Selection Sort, Insertion Sort
- **Searching:** Binary Search, Linear Search
- **Recursion:** Factorial, Fibonacci
- **Data Structures:** Two Sum, Array operations
- **Custom:** Any valid Python code!

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `→` | Next step |
| `←` | Previous step |
| `Home` | First step |
| `End` | Last step |
| `Ctrl+Enter` | Analyze code |

---

## 🔧 Configuration

### AI Explanation Styles

Choose from three explanation styles:

- **Beginner** 🎓 - Simple, detailed explanations
- **Intermediate** 📚 - Balanced technical depth
- **Advanced** 🔬 - Concise, expert-level

### OpenAI Integration (Optional)

For enhanced AI explanations, add your OpenAI API key:

```bash
# In .env file
OPENAI_API_KEY=sk-your-key-here
```

Without OpenAI, the app uses a local template-based explanation engine.

---

## 🛡️ Security

- **No code execution** - Uses AST parsing only
- **Safe analysis** - Flags potentially unsafe constructs
- **Input validation** - All inputs are validated
- **CORS configured** - Secure API access

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - feel free to use in your projects!

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI inspired by modern code editors
- AI explanations powered by template engine + OpenAI

---

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Contact: [your-email@example.com]

---

<p align="center">
  Made with ❤️ for developers and students
</p>
