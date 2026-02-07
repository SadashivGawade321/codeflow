"""
CodeFlow AI - Step Generator Module
Converts parsed AST into step-by-step execution model for visualization.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from .ast_parser import ParsedNode, ParseResult, NodeType


class StepType(str, Enum):
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


@dataclass
class VisualizationHint:
    """Hints for frontend visualization."""
    highlight_lines: List[int] = field(default_factory=list)
    highlight_variables: List[str] = field(default_factory=list)
    array_indices: List[int] = field(default_factory=list)
    pointers: Dict[str, int] = field(default_factory=dict)
    animation_type: str = "default"
    focus_element: Optional[str] = None


@dataclass
class ExecutionStep:
    """Represents a single step in code execution."""
    step_id: int
    step_type: StepType
    line_number: int
    code_snippet: str
    description: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    visualization: VisualizationHint
    ai_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "description": self.description,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "visualization": asdict(self.visualization),
            "ai_context": self.ai_context
        }


class StepGenerator:
    """
    Generates step-by-step execution model from parsed AST.
    Simulates execution for visualization purposes.
    """
    
    def __init__(self):
        self.steps: List[ExecutionStep] = []
        self.step_counter: int = 0
        self.variables: Dict[str, Any] = {}
        self.call_stack: List[str] = []
        self.loop_stack: List[Dict[str, Any]] = []
        self.source_lines: List[str] = []
    
    def generate(self, parse_result: ParseResult, input_values: Optional[Dict[str, Any]] = None) -> List[ExecutionStep]:
        """
        Generate execution steps from parsed code.
        
        Args:
            parse_result: Result from AST parser
            input_values: Optional initial variable values for simulation
            
        Returns:
            List of ExecutionStep objects
        """
        self.steps = []
        self.step_counter = 0
        self.variables = input_values.copy() if input_values else {}
        self.call_stack = []
        self.loop_stack = []
        self.source_lines = parse_result.source_lines
        
        if not parse_result.success:
            return []
        
        # Add initialization step if we have input values
        if input_values:
            self._add_init_step(input_values)
        
        # Process all nodes
        for node in parse_result.nodes:
            self._process_node(node)
        
        return self.steps
    
    def _add_init_step(self, values: Dict[str, Any]) -> None:
        """Add initialization step showing initial state."""
        var_descriptions = [f"{k} = {repr(v)}" for k, v in values.items()]
        
        self._add_step(
            step_type=StepType.INITIALIZE,
            line_number=0,
            code_snippet="# Initial State",
            description=f"Initialize variables: {', '.join(var_descriptions)}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_variables=list(values.keys()),
                animation_type="fade_in"
            ),
            ai_context={
                "action": "initialization",
                "variables": values,
                "purpose": "Set up initial values before execution begins"
            }
        )
    
    def _add_step(
        self,
        step_type: StepType,
        line_number: int,
        code_snippet: str,
        description: str,
        state_changes: Dict[str, Any],
        visualization: VisualizationHint,
        ai_context: Dict[str, Any]
    ) -> None:
        """Add a new execution step."""
        self.step_counter += 1
        
        state_before = self.variables.copy()
        
        # Apply state changes
        for key, value in state_changes.items():
            self.variables[key] = value
        
        step = ExecutionStep(
            step_id=self.step_counter,
            step_type=step_type,
            line_number=line_number,
            code_snippet=code_snippet,
            description=description,
            state_before=state_before,
            state_after=self.variables.copy(),
            visualization=visualization,
            ai_context=ai_context
        )
        
        self.steps.append(step)
    
    def _process_node(self, node: ParsedNode) -> None:
        """Process a single parsed node and generate steps."""
        
        if node.node_type == NodeType.ASSIGNMENT:
            self._process_assignment(node)
        
        elif node.node_type == NodeType.FUNCTION_DEF:
            self._process_function_def(node)
        
        elif node.node_type == NodeType.IF_STATEMENT:
            self._process_if(node)
        
        elif node.node_type == NodeType.FOR_LOOP:
            self._process_for(node)
        
        elif node.node_type == NodeType.WHILE_LOOP:
            self._process_while(node)
        
        elif node.node_type == NodeType.FUNCTION_CALL:
            self._process_call(node)
        
        elif node.node_type == NodeType.RETURN:
            self._process_return(node)
        
        elif node.node_type == NodeType.BREAK:
            self._process_break(node)
        
        elif node.node_type == NodeType.CONTINUE:
            self._process_continue(node)
    
    def _process_assignment(self, node: ParsedNode) -> None:
        """Generate steps for assignment."""
        targets = node.details.get("targets", [])
        value = node.details.get("value", "?")
        is_augmented = node.details.get("augmented", False)
        
        # Check if this is an array/list update
        is_array_update = any("[" in t for t in targets)
        
        # Determine step type
        if is_array_update:
            step_type = StepType.ARRAY_UPDATE
            animation = "highlight_index"
        else:
            step_type = StepType.ASSIGN
            animation = "value_change"
        
        # Simulate value evaluation
        evaluated_value = self._simulate_value(value)
        state_changes = {}
        
        for target in targets:
            if "[" in target:
                # Array update - parse index (base name extracted for potential future use)
                _ = target.split("[")[0]  # noqa: F841 - reserved for future use
                state_changes[target] = evaluated_value
            else:
                state_changes[target] = evaluated_value
        
        description = f"{'Update' if is_augmented else 'Assign'} {', '.join(targets)} = {value}"
        if is_augmented:
            op = node.details.get("operator", "=")
            description = f"Update {targets[0]} {op} {value}"
        
        self._add_step(
            step_type=step_type,
            line_number=node.line_number,
            code_snippet=node.code_snippet,
            description=description,
            state_changes=state_changes,
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                highlight_variables=targets,
                animation_type=animation
            ),
            ai_context={
                "action": "assignment",
                "targets": targets,
                "value": value,
                "evaluated": evaluated_value,
                "is_augmented": is_augmented,
                "value_type": node.details.get("value_type", "unknown")
            }
        )
    
    def _process_function_def(self, node: ParsedNode) -> None:
        """Generate steps for function definition."""
        func_name = node.details.get("name", "unknown")
        args = node.details.get("arguments", [])
        
        self._add_step(
            step_type=StepType.FUNCTION_ENTER,
            line_number=node.line_number,
            code_snippet=f"def {func_name}({', '.join(args)}):",
            description=f"Define function '{func_name}' with parameters: {', '.join(args) or 'none'}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="function_define"
            ),
            ai_context={
                "action": "function_definition",
                "function_name": func_name,
                "parameters": args,
                "has_return": node.details.get("has_return", False)
            }
        )
        
        # Process function body
        self.call_stack.append(func_name)
        for child in node.children:
            self._process_node(child)
        if self.call_stack:
            self.call_stack.pop()
    
    def _process_if(self, node: ParsedNode) -> None:
        """Generate steps for if statement."""
        condition = node.details.get("condition", "?")
        condition_analysis = node.details.get("condition_analysis", {})
        
        # Simulate condition evaluation
        condition_result = self._simulate_condition(condition)
        
        self._add_step(
            step_type=StepType.EVALUATE_CONDITION,
            line_number=node.line_number,
            code_snippet=f"if {condition}:",
            description=f"Evaluate condition: {condition} → {condition_result}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="condition_check"
            ),
            ai_context={
                "action": "condition_evaluation",
                "condition": condition,
                "condition_analysis": condition_analysis,
                "result": condition_result,
                "reason": self._get_condition_reason(condition, condition_result)
            }
        )
        
        if condition_result:
            self._add_step(
                step_type=StepType.ENTER_BRANCH,
                line_number=node.line_number + 1,
                code_snippet="# Enter if block",
                description="Condition is True, entering if block",
                state_changes={},
                visualization=VisualizationHint(
                    highlight_lines=list(range(node.line_number, node.end_line + 1)),
                    animation_type="branch_enter"
                ),
                ai_context={
                    "action": "enter_branch",
                    "branch_type": "if",
                    "condition_was": condition_result
                }
            )
            
            # Process if body (only non-else children)
            for child in node.children:
                if child.node_type != NodeType.IF_STATEMENT or child.line_number == node.line_number:
                    continue
                self._process_node(child)
        else:
            self._add_step(
                step_type=StepType.SKIP_BRANCH,
                line_number=node.line_number,
                code_snippet="# Skip if block",
                description="Condition is False, skipping if block",
                state_changes={},
                visualization=VisualizationHint(
                    highlight_lines=[node.line_number],
                    animation_type="branch_skip"
                ),
                ai_context={
                    "action": "skip_branch",
                    "branch_type": "if",
                    "condition_was": condition_result
                }
            )
    
    def _process_for(self, node: ParsedNode) -> None:
        """Generate steps for for loop."""
        loop_var = node.details.get("loop_variable", "i")
        iterable = node.details.get("iterable", {})
        
        # Simulate iterable
        values = self._simulate_iterable(iterable)
        
        self._add_step(
            step_type=StepType.START_LOOP,
            line_number=node.line_number,
            code_snippet=node.code_snippet.split('\n')[0],
            description=f"Start for loop: {loop_var} will iterate over {len(values)} elements",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="loop_start"
            ),
            ai_context={
                "action": "loop_start",
                "loop_type": "for",
                "loop_variable": loop_var,
                "iterable_type": iterable.get("type", "unknown"),
                "total_iterations": len(values)
            }
        )
        
        # Generate steps for each iteration (limit to prevent infinite loops)
        max_iterations = min(len(values), 10)  # Cap at 10 for visualization
        
        for i, value in enumerate(values[:max_iterations]):
            self.loop_stack.append({"type": "for", "var": loop_var, "iteration": i})
            
            self._add_step(
                step_type=StepType.LOOP_ITERATION,
                line_number=node.line_number,
                code_snippet=f"# Iteration {i + 1}",
                description=f"Loop iteration {i + 1}: {loop_var} = {value}",
                state_changes={loop_var: value},
                visualization=VisualizationHint(
                    highlight_lines=[node.line_number],
                    highlight_variables=[loop_var],
                    array_indices=[i] if isinstance(values, list) else [],
                    animation_type="iteration"
                ),
                ai_context={
                    "action": "loop_iteration",
                    "iteration_number": i + 1,
                    "loop_variable": loop_var,
                    "current_value": value,
                    "remaining": len(values) - i - 1
                }
            )
            
            # Process loop body
            for child in node.children:
                self._process_node(child)
            
            if self.loop_stack:
                self.loop_stack.pop()
        
        if len(values) > max_iterations:
            self._add_step(
                step_type=StepType.LOOP_ITERATION,
                line_number=node.line_number,
                code_snippet=f"# ... {len(values) - max_iterations} more iterations",
                description=f"Continuing for {len(values) - max_iterations} more iterations...",
                state_changes={},
                visualization=VisualizationHint(animation_type="iteration_skip"),
                ai_context={"action": "iterations_skipped", "count": len(values) - max_iterations}
            )
        
        self._add_step(
            step_type=StepType.END_LOOP,
            line_number=node.end_line,
            code_snippet="# End for loop",
            description=f"For loop completed after {len(values)} iterations",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.end_line],
                animation_type="loop_end"
            ),
            ai_context={
                "action": "loop_end",
                "total_iterations": len(values),
                "final_value": values[-1] if values else None
            }
        )
    
    def _process_while(self, node: ParsedNode) -> None:
        """Generate steps for while loop."""
        condition = node.details.get("condition", "?")
        
        self._add_step(
            step_type=StepType.START_LOOP,
            line_number=node.line_number,
            code_snippet=node.code_snippet.split('\n')[0],
            description=f"Start while loop with condition: {condition}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="loop_start"
            ),
            ai_context={
                "action": "loop_start",
                "loop_type": "while",
                "condition": condition
            }
        )
        
        # Simulate a few iterations
        iteration = 0
        max_iterations = 5  # Prevent infinite visualization
        
        while iteration < max_iterations:
            condition_result = self._simulate_condition(condition)
            
            self._add_step(
                step_type=StepType.EVALUATE_CONDITION,
                line_number=node.line_number,
                code_snippet=f"while {condition}:",
                description=f"Check condition: {condition} → {condition_result}",
                state_changes={},
                visualization=VisualizationHint(
                    highlight_lines=[node.line_number],
                    animation_type="condition_check"
                ),
                ai_context={
                    "action": "condition_check",
                    "iteration": iteration + 1,
                    "condition": condition,
                    "result": condition_result
                }
            )
            
            if not condition_result:
                break
            
            iteration += 1
            self.loop_stack.append({"type": "while", "iteration": iteration})
            
            self._add_step(
                step_type=StepType.LOOP_ITERATION,
                line_number=node.line_number,
                code_snippet=f"# Iteration {iteration}",
                description=f"While loop iteration {iteration}",
                state_changes={},
                visualization=VisualizationHint(
                    highlight_lines=[node.line_number],
                    animation_type="iteration"
                ),
                ai_context={
                    "action": "loop_iteration",
                    "iteration_number": iteration
                }
            )
            
            for child in node.children:
                self._process_node(child)
            
            if self.loop_stack:
                self.loop_stack.pop()
            
            # For simulation, assume condition eventually becomes false
            if iteration >= 3:
                break
        
        self._add_step(
            step_type=StepType.END_LOOP,
            line_number=node.end_line,
            code_snippet="# End while loop",
            description="While loop completed",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.end_line],
                animation_type="loop_end"
            ),
            ai_context={"action": "loop_end", "iterations_shown": iteration}
        )
    
    def _process_call(self, node: ParsedNode) -> None:
        """Generate steps for function call."""
        func_name = node.details.get("function", "unknown")
        args = node.details.get("arguments", [])
        
        self._add_step(
            step_type=StepType.FUNCTION_CALL,
            line_number=node.line_number,
            code_snippet=node.code_snippet,
            description=f"Call function '{func_name}' with arguments: {', '.join(args) or 'none'}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="function_call"
            ),
            ai_context={
                "action": "function_call",
                "function": func_name,
                "arguments": args,
                "is_builtin": node.details.get("is_builtin", False)
            }
        )
    
    def _process_return(self, node: ParsedNode) -> None:
        """Generate steps for return statement."""
        value = node.details.get("value")
        
        self._add_step(
            step_type=StepType.RETURN,
            line_number=node.line_number,
            code_snippet=node.code_snippet,
            description=f"Return {value if value else 'None'}",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="return"
            ),
            ai_context={
                "action": "return",
                "return_value": value,
                "from_function": self.call_stack[-1] if self.call_stack else "main"
            }
        )
    
    def _process_break(self, node: ParsedNode) -> None:
        """Generate steps for break statement."""
        self._add_step(
            step_type=StepType.BREAK,
            line_number=node.line_number,
            code_snippet="break",
            description="Break out of current loop",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="break"
            ),
            ai_context={
                "action": "break",
                "loop_info": self.loop_stack[-1] if self.loop_stack else None
            }
        )
    
    def _process_continue(self, node: ParsedNode) -> None:
        """Generate steps for continue statement."""
        self._add_step(
            step_type=StepType.CONTINUE,
            line_number=node.line_number,
            code_snippet="continue",
            description="Skip to next loop iteration",
            state_changes={},
            visualization=VisualizationHint(
                highlight_lines=[node.line_number],
                animation_type="continue"
            ),
            ai_context={
                "action": "continue",
                "loop_info": self.loop_stack[-1] if self.loop_stack else None
            }
        )
    
    def _simulate_value(self, value: str) -> Any:
        """Simulate value evaluation (simplified)."""
        # Try to evaluate simple literals safely
        try:
            import ast as ast_module
            return ast_module.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            # If not a literal, check if it's a known variable
            if value in self.variables:
                return self.variables[value]
            return value
    
    def _simulate_condition(self, condition: str) -> bool:
        """Simulate condition evaluation (simplified)."""
        # Simple pattern matching for common conditions
        try:
            # Handle simple comparisons
            for op in ['<=', '>=', '==', '!=', '<', '>']:
                if op in condition:
                    parts = condition.split(op)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()
                        left_val = self.variables.get(left, left)
                        right_val = self.variables.get(right, right)
                        # Try to compare
                        try:
                            import ast as ast_module
                            left_val = ast_module.literal_eval(str(left_val))
                            right_val = ast_module.literal_eval(str(right_val))
                        except (ValueError, SyntaxError):
                            pass
                        # Return based on step count for simulation
                        return len(self.steps) % 2 == 0
            return len(self.steps) % 2 == 0
        except (ValueError, TypeError, AttributeError):
            # Default to alternating for simulation
            return len(self.steps) % 2 == 0
    
    def _simulate_iterable(self, iterable: Dict[str, Any]) -> List[Any]:
        """Simulate iterable values."""
        iter_type = iterable.get("type", "unknown")
        
        if iter_type == "range":
            args = iterable.get("args", ["5"])
            try:
                parsed_args = [int(a) if a.isdigit() else 5 for a in args]
                return list(range(*parsed_args))
            except (ValueError, TypeError):
                return list(range(5))
        
        elif iter_type == "variable":
            var_name = iterable.get("name", "")
            if var_name in self.variables:
                val = self.variables[var_name]
                if isinstance(val, (list, tuple, str)):
                    return list(val)
            return [0, 1, 2, 3, 4]
        
        elif iter_type == "list_literal":
            return iterable.get("elements", [])
        
        return [0, 1, 2, 3, 4]
    
    def _get_condition_reason(self, condition: str, result: bool) -> str:
        """Generate human-readable reason for condition result."""
        if result:
            return f"The condition '{condition}' evaluates to True"
        else:
            return f"The condition '{condition}' evaluates to False"


def generate_steps(parse_result: ParseResult, input_values: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Convenience function to generate steps as JSON-serializable dicts."""
    generator = StepGenerator()
    steps = generator.generate(parse_result, input_values)
    return [step.to_dict() for step in steps]


def generate_steps_json(parse_result: ParseResult, input_values: Optional[Dict[str, Any]] = None) -> str:
    """Generate steps as JSON string."""
    steps = generate_steps(parse_result, input_values)
    return json.dumps(steps, indent=2)
