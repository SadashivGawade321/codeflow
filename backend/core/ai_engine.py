"""
CodeFlow AI - AI Explanation Engine
Generates natural language explanations for code execution steps.
Supports Google AI (Gemini) and Groq integration with caching and local fallback.
"""

import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Google AI imports
try:
    import google.generativeai as genai  # type: ignore[import-untyped]
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    GOOGLE_AI_AVAILABLE = False

# Groq AI imports (free with generous limits)
try:
    from groq import Groq  # type: ignore[import-untyped]
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore[assignment]
    GROQ_AVAILABLE = False

# Simple in-memory cache for AI responses
_response_cache: Dict[str, Any] = {}
_rate_limited = False
_rate_limit_reset_time = 0

# Request throttling - track request timestamps
_request_timestamps: deque = deque(maxlen=100)
MAX_REQUESTS_PER_MINUTE = 25  # Higher limit for new API key


class ExplanationStyle(str, Enum):
    """Style of AI explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class AIExplanation:
    """AI-generated explanation for a code step."""
    step_id: int
    summary: str
    detailed_explanation: str
    why_this_happens: str
    what_to_notice: List[str]
    common_mistakes: List[str]
    complexity_note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "why_this_happens": self.why_this_happens,
            "what_to_notice": self.what_to_notice,
            "common_mistakes": self.common_mistakes,
            "complexity_note": self.complexity_note
        }


class LocalAIEngine:
    """
    Local AI explanation engine using templates and rules.
    Falls back when no external API is available.
    """
    
    def __init__(self, style: ExplanationStyle = ExplanationStyle.BEGINNER):
        self.style = style
        self.explanation_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load explanation templates for different step types."""
        return {
            "initialize": {
                "summary": "Setting up initial values",
                "template": "Before the code runs, we set up the starting values: {variables}. These are the inputs our algorithm will work with.",
                "why": "Initialization ensures all variables have known starting values before we begin processing.",
                "notice": ["Check the data types", "Note the initial values", "Understand what each variable represents"]
            },
            "assign": {
                "summary": "Storing a value in a variable",
                "template": "We're assigning {value} to {target}. This {action} the variable.",
                "why": "Assignment stores data for later use. The variable {target} now holds {value}.",
                "notice": ["The old value is replaced", "The new value is now accessible via the variable name"]
            },
            "evaluate_condition": {
                "summary": "Checking if a condition is true or false",
                "template": "We're evaluating the condition '{condition}'. {result_explanation}",
                "why": "Conditions control program flow. Based on whether this is True or False, different code paths execute.",
                "notice": ["What values are being compared", "What operator is used (==, <, >, etc.)", "The boolean result"]
            },
            "enter_branch": {
                "summary": "Entering a code block because condition was true",
                "template": "The condition was True, so we enter the {branch_type} block.",
                "why": "When a condition evaluates to True, the code inside that branch executes.",
                "notice": ["Only this branch executes", "The else branch (if any) is skipped"]
            },
            "skip_branch": {
                "summary": "Skipping a code block because condition was false",
                "template": "The condition was False, so we skip the {branch_type} block.",
                "why": "When a condition evaluates to False, the code inside is not executed.",
                "notice": ["This code is not run", "Execution continues after the block or in an else branch"]
            },
            "start_loop": {
                "summary": "Beginning a loop",
                "template": "Starting a {loop_type} loop. {loop_description}",
                "why": "Loops allow us to repeat code multiple times without writing it repeatedly.",
                "notice": ["The loop variable", "How many iterations will occur", "The loop condition"]
            },
            "loop_iteration": {
                "summary": "Executing one cycle of the loop",
                "template": "Iteration {iteration}: {loop_var} = {current_value}. {remaining_info}",
                "why": "Each iteration processes one element or one step of the repetition.",
                "notice": ["Current loop variable value", "How state changes each iteration", "Progress toward completion"]
            },
            "end_loop": {
                "summary": "Loop has finished all iterations",
                "template": "The loop has completed after {iterations} iterations.",
                "why": "The loop ends when its condition becomes False or all elements are processed.",
                "notice": ["Final state of variables", "Total work done", "What changed from start to end"]
            },
            "function_call": {
                "summary": "Calling a function",
                "template": "Calling function '{function}' with arguments: {arguments}.",
                "why": "Functions encapsulate reusable logic. Calling them executes that logic with given inputs.",
                "notice": ["What arguments are passed", "What the function is expected to do", "The return value"]
            },
            "function_enter": {
                "summary": "Defining a new function",
                "template": "Defining function '{name}' with parameters: {params}.",
                "why": "Function definitions create reusable blocks of code that can be called later.",
                "notice": ["The function name", "What parameters it accepts", "What it will do when called"]
            },
            "return": {
                "summary": "Returning a value from function",
                "template": "Returning {value} from the function.",
                "why": "Return statements send a value back to where the function was called.",
                "notice": ["What value is returned", "This ends the function execution", "The caller receives this value"]
            },
            "array_update": {
                "summary": "Modifying an element in a list/array",
                "template": "Updating {target} to {value}. This changes the element at that index.",
                "why": "Array/list elements can be modified individually by their index.",
                "notice": ["Which index is being modified", "The new value", "The array state before and after"]
            },
            "swap": {
                "summary": "Swapping two values",
                "template": "Swapping values: {item1} ↔ {item2}.",
                "why": "Swapping exchanges two values, commonly used in sorting algorithms.",
                "notice": ["Both values change simultaneously", "This is often done with a temporary variable", "The positions are now reversed"]
            },
            "break": {
                "summary": "Exiting the loop early",
                "template": "Breaking out of the loop.",
                "why": "Break immediately exits the loop, skipping remaining iterations.",
                "notice": ["The loop ends immediately", "Code after the loop continues", "This is often used when a condition is met"]
            },
            "continue": {
                "summary": "Skipping to the next iteration",
                "template": "Continuing to the next loop iteration.",
                "why": "Continue skips the rest of the current iteration and moves to the next one.",
                "notice": ["Remaining code in this iteration is skipped", "The loop continues with the next value", "The loop does not exit"]
            }
        }
    
    def explain_step(self, step: Dict[str, Any]) -> AIExplanation:
        """Generate explanation for a single execution step."""
        step_type = step.get("step_type", "unknown")
        ai_context = step.get("ai_context", {})
        template_data = self.explanation_templates.get(step_type, {})
        
        # Generate explanation based on step type
        if step_type == "initialize":
            return self._explain_initialize(step, ai_context, template_data)
        elif step_type == "assign":
            return self._explain_assign(step, ai_context, template_data)
        elif step_type == "evaluate_condition":
            return self._explain_condition(step, ai_context, template_data)
        elif step_type in ["enter_branch", "skip_branch"]:
            return self._explain_branch(step, ai_context, template_data)
        elif step_type == "start_loop":
            return self._explain_loop_start(step, ai_context, template_data)
        elif step_type == "loop_iteration":
            return self._explain_iteration(step, ai_context, template_data)
        elif step_type == "end_loop":
            return self._explain_loop_end(step, ai_context, template_data)
        elif step_type == "function_call":
            return self._explain_function_call(step, ai_context, template_data)
        elif step_type == "function_enter":
            return self._explain_function_def(step, ai_context, template_data)
        elif step_type == "return":
            return self._explain_return(step, ai_context, template_data)
        elif step_type == "array_update":
            return self._explain_array_update(step, ai_context, template_data)
        elif step_type in ["break", "continue"]:
            return self._explain_control_flow(step, ai_context, template_data)
        else:
            return self._explain_generic(step, ai_context)
    
    def _explain_initialize(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        variables = context.get("variables", {})
        var_list = ", ".join([f"{k}={v}" for k, v in variables.items()])
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=template.get("summary", "Initialization"),
            detailed_explanation=template.get("template", "").format(variables=var_list),
            why_this_happens=template.get("why", ""),
            what_to_notice=template.get("notice", []),
            common_mistakes=["Forgetting to initialize variables", "Using wrong initial values"],
            complexity_note="Initialization is O(1) for each variable"
        )
    
    def _explain_assign(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        targets = context.get("targets", ["variable"])
        value = context.get("value", "value")
        is_augmented = context.get("is_augmented", False)
        action = "updates" if is_augmented else "creates or overwrites"
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Assign {value} to {', '.join(targets)}",
            detailed_explanation=template.get("template", "").format(
                value=value, target=targets[0], action=action
            ),
            why_this_happens=template.get("why", "").format(
                target=targets[0], value=value
            ),
            what_to_notice=template.get("notice", []) + [
                f"The variable '{targets[0]}' now holds '{value}'"
            ],
            common_mistakes=[
                "Confusing = (assignment) with == (comparison)",
                "Overwriting important values accidentally"
            ],
            complexity_note="Assignment is O(1)"
        )
    
    def _explain_condition(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        condition = context.get("condition", "condition")
        result = context.get("result", True)
        result_explanation = f"This evaluates to {result}."
        
        # Add reasoning for the result
        analysis = context.get("condition_analysis", {})
        if analysis.get("type") == "comparison":
            left = analysis.get("left", "left")
            ops = analysis.get("operators", ["=="])
            comps = analysis.get("comparators", ["right"])
            result_explanation += f" Because {left} {ops[0]} {comps[0]} is {result}."
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Evaluate: {condition} → {result}",
            detailed_explanation=template.get("template", "").format(
                condition=condition, result_explanation=result_explanation
            ),
            why_this_happens=template.get("why", ""),
            what_to_notice=[
                f"The condition evaluates to {result}",
                "This determines which code path executes next"
            ] + template.get("notice", []),
            common_mistakes=[
                "Using = instead of == for comparison",
                "Off-by-one errors in comparisons",
                "Not considering edge cases"
            ],
            complexity_note="Condition evaluation is O(1)"
        )
    
    def _explain_branch(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        branch_type = context.get("branch_type", "if")
        # Note: step["step_type"] indicates if entering or skipping
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=template.get("summary", "Branch decision"),
            detailed_explanation=template.get("template", "").format(branch_type=branch_type),
            why_this_happens=template.get("why", ""),
            what_to_notice=template.get("notice", []),
            common_mistakes=[
                "Missing else clause for edge cases",
                "Not handling all possible conditions"
            ]
        )
    
    def _explain_loop_start(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        loop_type = context.get("loop_type", "for")
        loop_var = context.get("loop_variable", "i")
        total = context.get("total_iterations", "unknown")
        
        if loop_type == "for":
            loop_description = f"Variable '{loop_var}' will iterate {total} times."
        else:
            loop_description = "The loop will continue while the condition is True."
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Start {loop_type} loop",
            detailed_explanation=template.get("template", "").format(
                loop_type=loop_type, loop_description=loop_description
            ),
            why_this_happens=template.get("why", ""),
            what_to_notice=template.get("notice", []) + [
                f"Loop variable: {loop_var}",
                f"Expected iterations: {total}"
            ],
            common_mistakes=[
                "Off-by-one errors in loop bounds",
                "Infinite loops from wrong conditions",
                "Modifying loop variable inside the loop"
            ],
            complexity_note=f"Loop will run O({total}) times"
        )
    
    def _explain_iteration(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        iteration = context.get("iteration_number", 1)
        loop_var = context.get("loop_variable", "i")
        current = context.get("current_value", "value")
        remaining = context.get("remaining", 0)
        
        remaining_info = f"{remaining} iterations remaining." if remaining > 0 else "This is the last iteration."
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Iteration {iteration}: {loop_var} = {current}",
            detailed_explanation=template.get("template", "").format(
                iteration=iteration, loop_var=loop_var,
                current_value=current, remaining_info=remaining_info
            ),
            why_this_happens="Each iteration processes one element or step.",
            what_to_notice=[
                f"Current value: {loop_var} = {current}",
                f"Iteration: {iteration}",
                remaining_info
            ],
            common_mistakes=[
                "Accessing wrong index",
                "Not updating loop-dependent variables"
            ]
        )
    
    def _explain_loop_end(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        iterations = context.get("total_iterations", "several")
        
        return AIExplanation(
            step_id=step["step_id"],
            summary="Loop completed",
            detailed_explanation=template.get("template", "").format(iterations=iterations),
            why_this_happens=template.get("why", ""),
            what_to_notice=[
                f"Total iterations: {iterations}",
                "Check the final state of variables",
                "Loop has finished executing"
            ],
            common_mistakes=[
                "Forgetting to use loop results",
                "Not handling empty iteration case"
            ],
            complexity_note=f"Loop completed in O({iterations}) iterations"
        )
    
    def _explain_function_call(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        func = context.get("function", "function")
        args = context.get("arguments", [])
        is_builtin = context.get("is_builtin", False)
        
        args_str = ", ".join(str(a) for a in args) if args else "no arguments"
        
        extra_info = ""
        if is_builtin:
            extra_info = f" ('{func}' is a Python built-in function)"
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Call {func}({args_str})",
            detailed_explanation=template.get("template", "").format(
                function=func, arguments=args_str
            ) + extra_info,
            why_this_happens=template.get("why", ""),
            what_to_notice=template.get("notice", []) + [
                f"Function: {func}",
                f"Arguments: {args_str}"
            ],
            common_mistakes=[
                "Wrong number of arguments",
                "Wrong argument order",
                "Not handling return value"
            ]
        )
    
    def _explain_function_def(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        name = context.get("function_name", "function")
        params = context.get("parameters", [])
        params_str = ", ".join(params) if params else "none"
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Define function {name}",
            detailed_explanation=template.get("template", "").format(
                name=name, params=params_str
            ),
            why_this_happens=template.get("why", ""),
            what_to_notice=[
                f"Function name: {name}",
                f"Parameters: {params_str}",
                "This code doesn't run until the function is called"
            ],
            common_mistakes=[
                "Forgetting return statement",
                "Not handling all parameter cases",
                "Side effects in functions"
            ]
        )
    
    def _explain_return(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        value = context.get("return_value", "None")
        from_func = context.get("from_function", "function")
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Return {value}",
            detailed_explanation=template.get("template", "").format(value=value),
            why_this_happens=template.get("why", ""),
            what_to_notice=[
                f"Return value: {value}",
                f"Returning from: {from_func}",
                "Function execution ends here"
            ],
            common_mistakes=[
                "Forgetting to return a value",
                "Returning wrong type",
                "Code after return is unreachable"
            ]
        )
    
    def _explain_array_update(self, step: Dict, context: Dict, template: Dict) -> AIExplanation:
        targets = context.get("targets", ["array[i]"])
        value = context.get("value", "value")
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=f"Update {targets[0]} = {value}",
            detailed_explanation=template.get("template", "").format(
                target=targets[0], value=value
            ),
            why_this_happens=template.get("why", ""),
            what_to_notice=[
                f"Modified element: {targets[0]}",
                f"New value: {value}",
                "Array is modified in-place"
            ],
            common_mistakes=[
                "Index out of bounds",
                "Off-by-one errors",
                "Modifying while iterating"
            ],
            complexity_note="Array access/update is O(1)"
        )
    
    def _explain_control_flow(self, step: Dict, _context: Dict, template: Dict) -> AIExplanation:
        action = step["step_type"]
        
        return AIExplanation(
            step_id=step["step_id"],
            summary=template.get("summary", f"{action} statement"),
            detailed_explanation=template.get("template", f"Executing {action}"),
            why_this_happens=template.get("why", ""),
            what_to_notice=template.get("notice", []),
            common_mistakes=[
                "Using break when continue is needed",
                "Breaking out of wrong loop level"
            ]
        )
    
    def _explain_generic(self, step: Dict, _context: Dict) -> AIExplanation:
        return AIExplanation(
            step_id=step["step_id"],
            summary=step.get("description", "Code execution step"),
            detailed_explanation=f"Executing: {step.get('code_snippet', 'code')}",
            why_this_happens="This is part of the program's logic flow.",
            what_to_notice=["Observe how state changes", "Track variable values"],
            common_mistakes=[]
        )
    
    def explain_all_steps(self, steps: List[Dict[str, Any]]) -> List[AIExplanation]:
        """Generate explanations for all steps."""
        return [self.explain_step(step) for step in steps]
    
    def get_complexity_analysis(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall time and space complexity from steps."""
        loop_count = sum(1 for s in steps if s["step_type"] == "start_loop")
        max_iterations = 0
        
        # Simple heuristic analysis
        for step in steps:
            if step["step_type"] == "loop_iteration":
                iteration = step.get("ai_context", {}).get("iteration_number", 1)
                max_iterations = max(max_iterations, iteration)
        
        # Determine complexity
        if loop_count == 0:
            time_complexity = "O(1)"
            complexity_reason = "No loops, constant time operations only"
        elif loop_count == 1:
            time_complexity = "O(n)"
            complexity_reason = "Single loop iterating through elements"
        elif loop_count == 2:
            time_complexity = "O(n) or O(n²)"
            complexity_reason = "Two loops - sequential is O(n), nested is O(n²)"
        else:
            time_complexity = "O(n^k)"
            complexity_reason = f"Multiple loops ({loop_count} detected)"
        
        return {
            "time_complexity": time_complexity,
            "space_complexity": "O(1) auxiliary",  # Simplified
            "loop_count": loop_count,
            "max_iterations_observed": max_iterations,
            "explanation": complexity_reason
        }


class OpenAIEngine:
    """
    AI explanation engine using OpenAI API.
    Provides more sophisticated explanations.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)
    
    async def explain_step_async(self, step: Dict[str, Any], code_context: str = "") -> Dict[str, Any]:
        """Generate AI explanation using OpenAI API (async)."""
        if not self.is_available():
            return {"error": "OpenAI API key not configured"}
        
        prompt = self._build_prompt(step, code_context)
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert programming tutor. Explain code execution steps 
                        in a clear, beginner-friendly way. Focus on:
                        1. What is happening in this step
                        2. Why it's happening
                        3. What to notice/learn
                        Keep explanations concise but educational."""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        explanation = data["choices"][0]["message"]["content"]
                        return {
                            "success": True,
                            "explanation": explanation,
                            "model": self.model
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API error: {response.status}"
                        }
        
        except ImportError:
            return {"error": "aiohttp not installed for async requests"}
        except (ValueError, TypeError, KeyError, ConnectionError) as e:
            return {"error": str(e)}
    
    def _build_prompt(self, step: Dict[str, Any], code_context: str) -> str:
        """Build prompt for OpenAI API."""
        return f"""
Explain this code execution step to a beginner:

**Code Context:**
```python
{code_context}
```

**Current Step:**
- Step Type: {step.get('step_type')}
- Line: {step.get('line_number')}
- Code: {step.get('code_snippet')}
- Description: {step.get('description')}
- State Before: {json.dumps(step.get('state_before', {}))}
- State After: {json.dumps(step.get('state_after', {}))}

Provide:
1. A one-sentence summary
2. Detailed explanation (2-3 sentences)
3. Why this happens
4. What to notice

Format as JSON with keys: summary, detailed, why, notice (array)
"""


class GoogleAIEngine:
    """
    AI explanation engine using Google AI (Gemini) API.
    Provides intelligent code explanations using Gemini models.
    Uses gemini-2.0-flash-lite for better free tier limits.
    Includes caching, throttling, and local fallback for rate limits.
    Now includes Groq as a backup for when Gemini hits rate limits.
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 3  # seconds
    RATE_LIMIT_COOLDOWN = 120  # seconds to wait before trying API again after rate limit
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.model: Any = None
        self.groq_client: Any = None
        self.local_engine = LocalAIEngine()  # Fallback engine
        
        # Initialize Gemini
        if self.api_key and GOOGLE_AI_AVAILABLE and genai:
            try:
                genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]
                self.model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
            except (ValueError, TypeError, RuntimeError, AttributeError):
                self.model = None
        
        # Initialize Groq as backup (use environment variable)
        groq_key = os.getenv("GROQ_API_KEY")
        if GROQ_AVAILABLE and Groq and groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
            except Exception:
                self.groq_client = None
    
    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API as fallback."""
        if not self.groq_client:
            return None
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq error: {e}")
            return None
    
    def _get_cache_key(self, prefix: str, *args: Any) -> str:
        """Generate a cache key from arguments."""
        content = prefix + "|".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_throttled(self) -> bool:
        """Check if we should throttle requests."""
        global _request_timestamps
        now = time.time()
        # Remove timestamps older than 60 seconds
        while _request_timestamps and now - _request_timestamps[0] > 60:
            _request_timestamps.popleft()
        # Check if we've hit the limit
        return len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE
    
    def _record_request(self) -> None:
        """Record a request timestamp."""
        global _request_timestamps
        _request_timestamps.append(time.time())
    
    def _is_rate_limited(self) -> bool:
        """Check if we're currently rate limited."""
        global _rate_limited, _rate_limit_reset_time
        if _rate_limited and time.time() < _rate_limit_reset_time:
            return True
        if _rate_limited and time.time() >= _rate_limit_reset_time:
            _rate_limited = False
        return False
    
    def _set_rate_limited(self) -> None:
        """Mark as rate limited."""
        global _rate_limited, _rate_limit_reset_time
        _rate_limited = True
        _rate_limit_reset_time = time.time() + self.RATE_LIMIT_COOLDOWN
    
    def _call_with_retry(self, prompt: str) -> Any:
        """Call Gemini API with retry logic for rate limits."""
        if self._is_rate_limited():
            raise RuntimeError("Rate limited - using local fallback")
        
        # Check throttling
        if self._is_throttled():
            raise RuntimeError("Throttled - too many requests per minute")
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self._record_request()
                response = self.model.generate_content(prompt)
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "429" in str(e) or "quota" in error_str or "rate" in error_str or "resource" in error_str:
                    if attempt == self.MAX_RETRIES - 1:
                        self._set_rate_limited()
                    wait_time = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        raise last_error if last_error else RuntimeError("Max retries exceeded")
    
    def is_available(self) -> bool:
        """Check if Google AI API is available."""
        return bool(self.api_key and self.model and GOOGLE_AI_AVAILABLE)
    
    def explain_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Generate comprehensive explanation for code using Gemini with caching."""
        # Check cache first
        cache_key = self._get_cache_key("explain", code, language)
        if cache_key in _response_cache:
            return _response_cache[cache_key]
        
        # If rate limited or throttled, use local engine
        if self._is_rate_limited() or self._is_throttled() or not self.is_available():
            return self._local_explain_code(code, language)
        
        prompt = f"""You are an expert programming tutor. Analyze this {language} code and provide a comprehensive explanation.

**Code:**
```{language}
{code}
```

Provide a JSON response with:
{{
    "summary": "One sentence overview of what this code does",
    "detailed_explanation": "2-3 paragraphs explaining the code step by step",
    "algorithm_type": "The algorithm or pattern used (e.g., sorting, searching, recursion)",
    "time_complexity": "Big O notation for time",
    "space_complexity": "Big O notation for space",
    "key_concepts": ["list", "of", "key", "programming", "concepts"],
    "common_mistakes": ["list", "of", "common", "mistakes"],
    "optimization_tips": ["list", "of", "possible", "optimizations"],
    "related_problems": ["list", "of", "similar", "problems"]
}}

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self._call_with_retry(prompt)
            text = response.text.strip()
            
            # Try to extract JSON from the response
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            
            result = json.loads(text)
            result["success"] = True
            # Cache the result
            _response_cache[cache_key] = result
            return result
            
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            result = {
                "success": True,
                "summary": response.text[:200],
                "detailed_explanation": response.text,
                "algorithm_type": "Unknown",
                "time_complexity": "Unknown",
                "space_complexity": "Unknown",
                "key_concepts": [],
                "common_mistakes": [],
                "optimization_tips": [],
                "related_problems": []
            }
            _response_cache[cache_key] = result
            return result
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                # Try Groq as fallback before local engine
                if GROQ_AVAILABLE and self.groq_client:
                    try:
                        groq_response = self._call_groq(prompt)
                        if groq_response:
                            text = groq_response.strip()
                            if text.startswith("```"):
                                lines = text.split("\n")
                                text = "\n".join(lines[1:-1])
                            result = json.loads(text)
                            result["success"] = True
                            result["provider"] = "groq"
                            _response_cache[cache_key] = result
                            return result
                    except Exception:
                        pass
                # Fallback to local engine
                return self._local_explain_code(code, language)
            return {"error": error_msg, "success": False}
    
    def _local_explain_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Generate explanation using local AI engine (fallback)."""
        # Use local engine for basic analysis
        lines = code.strip().split('\n')
        
        # Detect algorithm type
        algo_type = "General algorithm"
        if "sort" in code.lower():
            algo_type = "Sorting algorithm"
        elif "search" in code.lower() or "find" in code.lower():
            algo_type = "Search algorithm"
        elif "def " in code and code.count("def ") > 0:
            if any(fn in code for fn in ["fibonacci", "factorial", "recursi"]):
                algo_type = "Recursive algorithm"
        
        # Estimate complexity
        loop_count = code.count("for ") + code.count("while ")
        nested = "for " in code and code.count("for ") > 1
        time_comp = "O(n²)" if nested else ("O(n)" if loop_count > 0 else "O(1)")
        
        return {
            "success": True,
            "summary": f"This {language} code contains {len(lines)} lines implementing a {algo_type.lower()}.",
            "detailed_explanation": f"The code performs operations typical of a {algo_type.lower()}. It contains {loop_count} loop(s) which affect its time complexity.",
            "algorithm_type": algo_type,
            "time_complexity": time_comp,
            "space_complexity": "O(n)" if "[]" in code or "list" in code.lower() else "O(1)",
            "key_concepts": ["loops", "variables", "conditionals"],
            "common_mistakes": ["Off-by-one errors", "Infinite loops", "Uninitialized variables"],
            "optimization_tips": ["Consider using built-in functions", "Avoid unnecessary iterations"],
            "related_problems": [],
            "note": "Using local analysis (API rate limit). Full AI analysis will resume shortly."
        }
    
    def answer_question(self, question: str, code: str, language: str = "python") -> str:
        """Answer a question about code using Gemini/Groq with caching."""
        # Check cache first
        cache_key = self._get_cache_key("answer", question, code, language)
        if cache_key in _response_cache:
            return _response_cache[cache_key]
        
        prompt = f"""You are an expert programming tutor helping students understand code.

**{language.title()} Code:**
```{language}
{code}
```

**Student Question:** {question}

Provide a clear, educational answer that:
1. Directly addresses the question
2. References specific parts of the code
3. Explains the underlying concepts
4. Uses simple language suitable for beginners

Keep your response concise but complete."""

        # Try Gemini first
        if self.is_available() and not self._is_rate_limited() and not self._is_throttled():
            try:
                response = self._call_with_retry(prompt)
                result = response.text
                _response_cache[cache_key] = result
                return result
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    pass  # Fall through to Groq
                else:
                    pass  # Fall through to Groq
        
        # Try Groq as fallback
        groq_result = self._call_groq(prompt)
        if groq_result:
            _response_cache[cache_key] = groq_result
            return groq_result
        
        # Local fallback
        return "I can help explain this code! The algorithm processes the input data step by step. Try asking about specific lines or concepts."
    
    def run_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Simulate code execution and return the expected output with caching."""
        # Check cache first
        cache_key = self._get_cache_key("run", code, language)
        if cache_key in _response_cache:
            return _response_cache[cache_key]
        
        prompt = f"""You are a code execution simulator. Execute this {language} code and show ONLY the output.

**Code:**
```{language}
{code}
```

**IMPORTANT INSTRUCTIONS:**
1. Show ONLY the exact output that would appear in a terminal/console
2. Do NOT explain the code
3. Do NOT add any commentary or markdown
4. If there are no print/output statements, respond with: No output
5. If there's a syntax error or runtime error, show the error message as it would appear
6. For each print/output statement, show the result on a new line
7. Show the output exactly as it would appear when running the code

Output:"""

        # Try Gemini first
        if self.is_available() and not self._is_rate_limited() and not self._is_throttled():
            try:
                response = self._call_with_retry(prompt)
                output = response.text.strip()
                
                if output.startswith("```"):
                    lines = output.split("\n")
                    output = "\n".join(lines[1:-1]) if len(lines) > 2 else output
                
                result = {"success": True, "output": output, "language": language}
                _response_cache[cache_key] = result
                return result
            except Exception:
                pass  # Fall through to Groq
        
        # Try Groq as fallback
        groq_result = self._call_groq(prompt)
        if groq_result:
            output = groq_result.strip()
            if output.startswith("```"):
                lines = output.split("\n")
                output = "\n".join(lines[1:-1]) if len(lines) > 2 else output
            result = {"success": True, "output": output, "language": language}
            _response_cache[cache_key] = result
            return result
        
        return {"success": False, "output": "Unable to run code - all AI services unavailable", "error": "no_service"}
    
    def explain_step(self, step: Dict[str, Any], code: str, language: str = "python") -> Dict[str, Any]:
        """Explain a specific execution step using Gemini."""
        if not self.is_available():
            return {"error": "Google AI not configured"}
        
        prompt = f"""Explain this code execution step to a programming student.

**{language.title()} Code:**
```{language}
{code}
```

**Current Step:**
- Step Type: {step.get('step_type')}
- Line: {step.get('line_number')}
- Code: {step.get('code_snippet')}
- Description: {step.get('description')}
- Variables Before: {json.dumps(step.get('state_before', {}))}
- Variables After: {json.dumps(step.get('state_after', {}))}

Respond with JSON:
{{
    "summary": "Brief one-sentence summary",
    "detailed_explanation": "Clear 2-3 sentence explanation",
    "why_this_happens": "Why this step is important",
    "what_to_notice": ["key point 1", "key point 2"],
    "common_mistakes": ["mistake 1", "mistake 2"]
}}

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except (ValueError, TypeError, RuntimeError, AttributeError, json.JSONDecodeError) as e:
            return {"error": str(e)}
    
    def generate_hints(self, code: str, language: str = "python") -> List[str]:
        """Generate learning hints for the code."""
        if not self.is_available():
            return []
        
        prompt = f"""Analyze this {language} code and provide 3-5 educational hints that would help a student understand it better.

```{language}
{code}
```

Return a JSON array of strings, like: ["hint 1", "hint 2", "hint 3"]
Return ONLY the JSON array, nothing else."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except (ValueError, TypeError, RuntimeError, AttributeError, json.JSONDecodeError):
            return []


class AIExplanationEngine:
    """
    Main AI engine that combines local and API-based explanations.
    Supports Google AI (Gemini), OpenAI, and local fallback.
    """
    
    def __init__(
        self,
        style: ExplanationStyle = ExplanationStyle.BEGINNER,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        use_openai: bool = False,
        use_google_ai: bool = True
    ):
        self.style = style
        self.local_engine = LocalAIEngine(style)
        self.openai_engine: Optional[OpenAIEngine] = OpenAIEngine(openai_api_key) if use_openai else None
        self.google_engine = GoogleAIEngine(google_api_key)
        self.use_openai: bool = use_openai and bool(self.openai_engine and self.openai_engine.is_available())
        self.use_google_ai: bool = use_google_ai and self.google_engine.is_available()
        self.current_language = "python"
    
    def set_language(self, language: str) -> None:
        """Set the current programming language."""
        self.current_language = language.lower()
    
    def is_ai_available(self) -> bool:
        """Check if any AI service is available."""
        return bool(self.use_google_ai or self.use_openai)
    
    def explain_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for a step."""
        explanation = self.local_engine.explain_step(step)
        return explanation.to_dict()
    
    def explain_all_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for all steps."""
        return [self.explain_step(step) for step in steps]
    
    def explain_code_with_ai(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Get comprehensive AI explanation for code."""
        if self.use_google_ai:
            return self.google_engine.explain_code(code, language)
        return {"error": "No AI service configured", "success": False}
    
    def run_code_simulation(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Simulate code execution using Gemini AI."""
        if self.use_google_ai:
            return self.google_engine.run_code(code, language)
        return {"success": False, "output": "AI service not configured"}
    
    def chat_with_gemini(self, question: str, code: str = "", language: str = "python") -> str:
        """Chat with Gemini AI about code."""
        if self.use_google_ai:
            return self.google_engine.answer_question(question, code, language)
        return "AI service not configured. Please set GOOGLE_AI_API_KEY."
    
    def get_complexity_analysis(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get complexity analysis."""
        return self.local_engine.get_complexity_analysis(steps)
    
    def answer_question(self, question: str, code: str, steps: List[Dict[str, Any]], language: str = "python") -> str:
        """Answer a user question about the code."""
        # Try Google AI first for intelligent responses
        if self.use_google_ai:
            return self.google_engine.answer_question(question, code, language)
        
        # Fallback to local pattern matching
        question_lower = question.lower()
        
        # Handle common question patterns
        if "why" in question_lower and "swap" in question_lower:
            return self._explain_swap(steps)
        elif "complexity" in question_lower or "big o" in question_lower:
            analysis = self.get_complexity_analysis(steps)
            return f"The time complexity is {analysis['time_complexity']}. {analysis['explanation']}"
        elif "loop" in question_lower:
            return self._explain_loops(steps)
        elif "condition" in question_lower or "if" in question_lower:
            return self._explain_conditions(steps)
        else:
            return self._general_answer(question, code, steps)
    
    def _explain_swap(self, steps: List[Dict[str, Any]]) -> str:
        """Explain swap operations in the code."""
        swap_steps = [s for s in steps if "swap" in s.get("description", "").lower()]
        if swap_steps:
            return "Swapping happens to rearrange elements. In sorting algorithms, we swap elements to put them in the correct order."
        return "No swap operations were detected in this code."
    
    def _explain_loops(self, steps: List[Dict[str, Any]]) -> str:
        """Explain loops in the code."""
        loop_starts = [s for s in steps if s["step_type"] == "start_loop"]
        if loop_starts:
            loop_info = loop_starts[0].get("ai_context", {})
            return f"This code uses a {loop_info.get('loop_type', 'for')} loop to iterate. The loop runs multiple times, processing each element or step sequentially."
        return "No loops were detected in this code."
    
    def _explain_conditions(self, steps: List[Dict[str, Any]]) -> str:
        """Explain conditions in the code."""
        condition_steps = [s for s in steps if s["step_type"] == "evaluate_condition"]
        if condition_steps:
            cond = condition_steps[0].get("ai_context", {})
            return f"The condition '{cond.get('condition', '?')}' is evaluated to decide which code path to take. It evaluated to {cond.get('result', 'True/False')}."
        return "No conditional statements were detected in this code."
    
    def _general_answer(self, _question: str, _code: str, steps: List[Dict[str, Any]]) -> str:
        """Provide a general answer based on the code and steps."""
        return f"""Based on the code analysis:
        
The code has {len(steps)} execution steps. It contains:
- {sum(1 for s in steps if 'loop' in s['step_type'])} loop-related steps
- {sum(1 for s in steps if 'condition' in s['step_type'])} condition evaluations
- {sum(1 for s in steps if s['step_type'] == 'assign')} variable assignments

For more specific information, try asking about:
- Time/space complexity
- Specific loops or conditions
- Why certain operations happen
"""


# Supported languages for multi-language analysis
SUPPORTED_LANGUAGES = {
    "python": {"extension": ".py", "name": "Python", "comment": "#"},
    "javascript": {"extension": ".js", "name": "JavaScript", "comment": "//"},
    "java": {"extension": ".java", "name": "Java", "comment": "//"},
    "cpp": {"extension": ".cpp", "name": "C++", "comment": "//"},
    "c": {"extension": ".c", "name": "C", "comment": "//"},
    "csharp": {"extension": ".cs", "name": "C#", "comment": "//"},
    "go": {"extension": ".go", "name": "Go", "comment": "//"},
    "rust": {"extension": ".rs", "name": "Rust", "comment": "//"},
    "typescript": {"extension": ".ts", "name": "TypeScript", "comment": "//"},
    "ruby": {"extension": ".rb", "name": "Ruby", "comment": "#"},
    "php": {"extension": ".php", "name": "PHP", "comment": "//"},
    "swift": {"extension": ".swift", "name": "Swift", "comment": "//"},
    "kotlin": {"extension": ".kt", "name": "Kotlin", "comment": "//"},
}


def get_supported_languages() -> Dict[str, Dict[str, str]]:
    """Get list of supported programming languages."""
    return SUPPORTED_LANGUAGES


# Convenience function
def create_ai_engine(
    style: str = "beginner",
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    use_openai: bool = False,
    use_google_ai: bool = True
) -> AIExplanationEngine:
    """Create an AI explanation engine with specified settings."""
    style_map = {
        "beginner": ExplanationStyle.BEGINNER,
        "intermediate": ExplanationStyle.INTERMEDIATE,
        "advanced": ExplanationStyle.ADVANCED
    }
    return AIExplanationEngine(
        style=style_map.get(style, ExplanationStyle.BEGINNER),
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        use_openai=use_openai,
        use_google_ai=use_google_ai
    )

