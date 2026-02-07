"""
CodeFlow AI - Pydantic Models
Shared data models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


class ExecutionStepType(str, Enum):
    """Types of execution steps."""
    INITIALIZE = "initialize"
    ASSIGN = "assign"
    EVALUATE_CONDITION = "evaluate_condition"
    ENTER_BRANCH = "enter_branch"
    SKIP_BRANCH = "skip_branch"
    START_LOOP = "start_loop"
    LOOP_ITERATION = "loop_iteration"
    END_LOOP = "end_loop"
    FUNCTION_CALL = "function_call"
    FUNCTION_ENTER = "function_enter"
    FUNCTION_EXIT = "function_exit"
    RETURN = "return"
    COMPARE = "compare"
    SWAP = "swap"
    ARRAY_ACCESS = "array_access"
    ARRAY_UPDATE = "array_update"
    BREAK = "break"
    CONTINUE = "continue"


class VisualizationHint(BaseModel):
    """Visualization hints for the frontend."""
    highlight_lines: List[int] = []
    highlight_variables: List[str] = []
    array_indices: List[int] = []
    pointers: Dict[str, int] = {}
    animation_type: str = "default"
    focus_element: Optional[str] = None


class ExecutionStep(BaseModel):
    """A single step in code execution."""
    step_id: int
    step_type: ExecutionStepType
    line_number: int
    code_snippet: str
    description: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    visualization: VisualizationHint
    ai_context: Dict[str, Any] = {}


class AIExplanation(BaseModel):
    """AI-generated explanation for a step."""
    step_id: int
    summary: str
    detailed_explanation: str
    why_this_happens: str
    what_to_notice: List[str]
    common_mistakes: List[str]
    complexity_note: Optional[str] = None


class ComplexityAnalysis(BaseModel):
    """Time and space complexity analysis."""
    time_complexity: str
    space_complexity: str
    loop_count: int
    max_iterations_observed: int
    explanation: str


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""
    code: str = Field(..., min_length=1, description="Python code to analyze")
    input_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial variable values"
    )
    style: str = Field(default="beginner", description="Explanation style")


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis."""
    success: bool
    source_lines: List[str] = []
    steps: List[ExecutionStep] = []
    explanations: List[AIExplanation] = []
    complexity: Optional[ComplexityAnalysis] = None
    error: Optional[str] = None


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    code: str
    question: str
    context: Optional[Dict[str, Any]] = None


class QuestionResponse(BaseModel):
    """Response model for questions."""
    success: bool
    question: str
    answer: str
    related_steps: List[int] = []


class CodeExample(BaseModel):
    """A code example for demonstration."""
    id: str
    name: str
    description: str
    code: str
    input_values: Optional[Dict[str, Any]] = None
    complexity: Optional[str] = None
    tags: List[str] = []
