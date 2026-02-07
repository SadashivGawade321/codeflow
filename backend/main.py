"""
CodeFlow AI - FastAPI Backend
Main application entry point with REST API endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
import os
import sys
import io
import contextlib
import traceback

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ast_parser import SafeASTParser, parse_code
from core.step_generator import StepGenerator, generate_steps
from core.ai_engine import create_ai_engine, get_supported_languages, GOOGLE_AI_AVAILABLE

# Import auth router
from auth import router as auth_router


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: Initialize AI engine
    application.state.ai_engine = create_ai_engine(style="beginner")
    yield
    # Shutdown: Cleanup completed


# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="CodeFlow AI",
    description="AI-powered code visualization and explanation platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Include auth router
app.include_router(auth_router)


# ============================================================================
# Pydantic Models
# ============================================================================

class CodeInput(BaseModel):
    """Input model for code analysis."""
    code: str = Field(..., description="Source code to analyze")
    language: str = Field(default="python", description="Programming language")
    input_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional initial variable values for simulation"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "arr = [5, 2, 8, 1]\nfor i in range(len(arr)):\n    print(arr[i])",
                "language": "python",
                "input_values": {"arr": [5, 2, 8, 1]}
            }
        }
    )


class QuestionInput(BaseModel):
    """Input model for asking questions about code."""
    code: str = Field(..., description="Source code")
    question: str = Field(..., description="Question about the code")
    language: str = Field(default="python", description="Programming language")
    steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Pre-computed execution steps"
    )


class AnalysisResponse(BaseModel):
    """Response model for code analysis."""
    success: bool
    source_lines: List[str]
    steps: List[Dict[str, Any]]
    explanations: List[Dict[str, Any]]
    complexity: Dict[str, Any]
    error: Optional[str] = None


class ExplanationResponse(BaseModel):
    """Response model for step explanation."""
    step_id: int
    explanation: Dict[str, Any]


# ============================================================================
# App State Helper
# ============================================================================

def get_ai_engine():
    """Get the AI engine from app state."""
    return app.state.ai_engine


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend application."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CodeFlow AI API", "docs": "/api/docs"}


@app.get("/auth")
async def auth_page():
    """Serve the authentication page."""
    auth_path = os.path.join(FRONTEND_DIR, "auth.html")
    if os.path.exists(auth_path):
        return FileResponse(auth_path)
    raise HTTPException(status_code=404, detail="Auth page not found")


@app.get("/styles.css")
async def serve_styles():
    """Serve the CSS file."""
    css_path = os.path.join(FRONTEND_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript file."""
    js_path = os.path.join(FRONTEND_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CodeFlow AI",
        "version": "1.0.0"
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_code(input_data: CodeInput):
    """
    Analyze Python code and generate visualization steps.
    
    This endpoint:
    1. Parses the code using AST (safely, without execution)
    2. Generates step-by-step execution model
    3. Provides AI explanations for each step
    4. Analyzes time/space complexity
    """
    try:
        # Parse the code
        parser = SafeASTParser()
        parse_result = parser.parse(input_data.code)
        
        if not parse_result.success:
            return AnalysisResponse(
                success=False,
                source_lines=parse_result.source_lines,
                steps=[],
                explanations=[],
                complexity={},
                error=parse_result.error
            )
        
        # Generate execution steps
        generator = StepGenerator()
        steps = generator.generate(parse_result, input_data.input_values)
        steps_dict = [step.to_dict() for step in steps]
        
        # Generate AI explanations
        ai = get_ai_engine()
        explanations = ai.explain_all_steps(steps_dict)
        
        # Analyze complexity
        complexity = ai.get_complexity_analysis(steps_dict)
        
        return AnalysisResponse(
            success=True,
            source_lines=parse_result.source_lines,
            steps=steps_dict,
            explanations=explanations,
            complexity=complexity
        )
    
    except (ValueError, TypeError, AttributeError) as e:
        return AnalysisResponse(
            success=False,
            source_lines=[],
            steps=[],
            explanations=[],
            complexity={},
            error=str(e)
        )


@app.post("/api/explain/step/{step_id}")
async def explain_step(step_id: int, input_data: CodeInput):
    """Get detailed explanation for a specific step."""
    try:
        # Parse and generate steps
        parse_result = parse_code(input_data.code)
        if not parse_result.success:
            raise HTTPException(status_code=400, detail=parse_result.error)
        
        steps = generate_steps(parse_result, input_data.input_values)
        
        # Find the specific step
        step = next((s for s in steps if s["step_id"] == step_id), None)
        if not step:
            raise HTTPException(status_code=404, detail=f"Step {step_id} not found")
        
        explanation = get_ai_engine().explain_step(step)
        
        return ExplanationResponse(
            step_id=step_id,
            explanation=explanation
        )
    
    except HTTPException as http_exc:
        raise http_exc
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/ask")
async def ask_question(input_data: QuestionInput):
    """
    Ask a question about the code.
    
    The AI will analyze the code and steps to provide an answer.
    Uses Google AI (Gemini) when available for intelligent responses.
    """
    try:
        # Generate steps if not provided (for Python only)
        steps = input_data.steps
        if not steps and input_data.language == "python":
            parse_result = parse_code(input_data.code)
            if parse_result.success:
                steps = generate_steps(parse_result)
            else:
                steps = []
        elif not steps:
            steps = []
        
        # Get AI answer with language context
        answer = get_ai_engine().answer_question(
            input_data.question,
            input_data.code,
            steps,
            input_data.language
        )
        
        return {
            "success": True,
            "question": input_data.question,
            "answer": answer,
            "language": input_data.language
        }
    
    except (ValueError, TypeError, AttributeError) as e:
        return {
            "success": False,
            "question": input_data.question,
            "answer": f"Error processing question: {str(e)}"
        }


@app.post("/api/complexity")
async def analyze_complexity(input_data: CodeInput):
    """Analyze time and space complexity of the code."""
    try:
        parse_result = parse_code(input_data.code)
        if not parse_result.success:
            raise HTTPException(status_code=400, detail=parse_result.error)
        
        steps = generate_steps(parse_result, input_data.input_values)
        complexity = get_ai_engine().get_complexity_analysis(steps)
        
        return {
            "success": True,
            "complexity": complexity
        }
    
    except HTTPException as http_exc:
        raise http_exc
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/languages")
async def get_languages():
    """Get list of supported programming languages."""
    languages = get_supported_languages()
    return {
        "languages": [
            {"id": lang_id, **lang_info}
            for lang_id, lang_info in languages.items()
        ],
        "default": "python"
    }


@app.post("/api/explain-code")
async def explain_code_with_ai(input_data: CodeInput):
    """
    Get comprehensive AI explanation for code in any language.
    Uses Google AI (Gemini) for intelligent analysis.
    """
    try:
        ai_engine = get_ai_engine()
        
        # Use Google AI for comprehensive explanation
        explanation = ai_engine.explain_code_with_ai(
            input_data.code,
            input_data.language
        )
        
        return {
            "success": explanation.get("success", False),
            "language": input_data.language,
            **explanation
        }
    
    except (ValueError, TypeError, AttributeError) as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/run-code")
async def run_code_with_ai(input_data: CodeInput):
    """
    Simulate code execution using Gemini AI.
    Returns the expected output of the code.
    """
    try:
        ai_engine = get_ai_engine()
        
        # Use Google AI to simulate code execution
        result = ai_engine.run_code_simulation(
            input_data.code,
            input_data.language
        )
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "output": f"Error: {str(e)}",
            "error": str(e)
        }


@app.post("/api/chat")
async def chat_with_ai(request: QuestionInput):
    """
    Chat with Gemini AI about code.
    Direct API endpoint for AI chat.
    """
    try:
        ai_engine = get_ai_engine()
        
        # Use Google AI for chat
        response = ai_engine.chat_with_gemini(
            request.question,
            request.code,
            request.language
        )
        
        return {
            "success": True,
            "answer": response
        }
    
    except Exception as e:
        return {
            "success": False,
            "answer": f"Sorry, I encountered an error: {str(e)}"
        }


@app.get("/api/ai-status")
async def get_ai_status():
    """Check if AI services are configured and available."""
    ai_engine = get_ai_engine()
    return {
        "google_ai_available": ai_engine.use_google_ai,
        "google_ai_library_installed": GOOGLE_AI_AVAILABLE,
        "openai_available": ai_engine.use_openai if ai_engine.openai_engine else False,
        "any_ai_available": ai_engine.is_ai_available(),
        "message": "Configure GOOGLE_AI_API_KEY environment variable to enable AI features" if not ai_engine.is_ai_available() else "AI is ready"
    }


@app.get("/api/examples")
async def get_examples():
    """Get example code snippets for demonstration."""
    examples = [
        {
            "id": "bubble_sort",
            "name": "Bubble Sort",
            "description": "Classic sorting algorithm",
            "code": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

result = bubble_sort([64, 34, 25, 12, 22, 11, 90])""",
            "input_values": {"arr": [64, 34, 25, 12, 22, 11, 90]}
        },
        {
            "id": "binary_search",
            "name": "Binary Search",
            "description": "Efficient search in sorted array",
            "code": """def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

result = binary_search([1, 3, 5, 7, 9, 11, 13], 7)""",
            "input_values": {"arr": [1, 3, 5, 7, 9, 11, 13], "target": 7}
        },
        {
            "id": "factorial",
            "name": "Factorial (Recursive)",
            "description": "Calculate factorial using recursion",
            "code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)""",
            "input_values": {"n": 5}
        },
        {
            "id": "fibonacci",
            "name": "Fibonacci Sequence",
            "description": "Generate Fibonacci numbers",
            "code": """def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

result = fibonacci(10)""",
            "input_values": {"n": 10}
        },
        {
            "id": "linear_search",
            "name": "Linear Search",
            "description": "Simple sequential search",
            "code": """def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

result = linear_search([4, 2, 7, 1, 9, 3], 7)""",
            "input_values": {"arr": [4, 2, 7, 1, 9, 3], "target": 7}
        },
        {
            "id": "two_sum",
            "name": "Two Sum",
            "description": "Find two numbers that add to target",
            "language": "python",
            "code": """def two_sum(nums, target):
    seen = {}
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in seen:
            return [seen[complement], i]
        seen[nums[i]] = i
    return []

result = two_sum([2, 7, 11, 15], 9)""",
            "input_values": {"nums": [2, 7, 11, 15], "target": 9}
        },
        # JavaScript Examples
        {
            "id": "bubble_sort_js",
            "name": "Bubble Sort",
            "description": "Classic sorting algorithm",
            "language": "javascript",
            "code": """function bubbleSort(arr) {
    const n = arr.length;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

const result = bubbleSort([64, 34, 25, 12, 22, 11, 90]);
console.log(result);""",
            "input_values": {}
        },
        {
            "id": "binary_search_js",
            "name": "Binary Search",
            "description": "Efficient search in sorted array",
            "language": "javascript",
            "code": """function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

const result = binarySearch([1, 3, 5, 7, 9, 11, 13], 7);
console.log(result);""",
            "input_values": {}
        },
        # Java Examples
        {
            "id": "bubble_sort_java",
            "name": "Bubble Sort",
            "description": "Classic sorting algorithm",
            "language": "java",
            "code": """public class BubbleSort {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    public static void main(String[] args) {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        bubbleSort(arr);
    }
}""",
            "input_values": {}
        },
        # C++ Examples
        {
            "id": "bubble_sort_cpp",
            "name": "Bubble Sort",
            "description": "Classic sorting algorithm",
            "language": "cpp",
            "code": """#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    bubbleSort(arr);
    return 0;
}""",
            "input_values": {}
        }
    ]
    
    return {"examples": examples}


@app.get("/api/settings")
async def get_settings():
    """Get current AI settings."""
    ai_engine = get_ai_engine()
    return {
        "style": "beginner",
        "available_styles": ["beginner", "intermediate", "advanced"],
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "google_ai_configured": ai_engine.use_google_ai,
        "languages": list(get_supported_languages().keys())
    }


@app.put("/api/settings")
async def update_settings(style: str = Query("beginner")):
    """Update AI settings."""
    app.state.ai_engine = create_ai_engine(style=style)
    return {"message": f"AI style updated to {style}"}


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
