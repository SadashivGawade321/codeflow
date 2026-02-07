"""
CodeFlow AI - Test Suite
Run with: python test_codeflow.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.ast_parser import parse_code
from backend.core.step_generator import generate_steps
from backend.core.ai_engine import create_ai_engine


class TestASTParser:
    """Tests for AST Parser"""
    
    def test_simple_assignment(self):
        code = "x = 5"
        result = parse_code(code)
        assert result.success
        assert len(result.nodes) == 1
        assert result.nodes[0].node_type.value == "assignment"
    
    def test_for_loop(self):
        code = """
for i in range(5):
    print(i)
"""
        result = parse_code(code)
        assert result.success
        assert any(n.node_type.value == "for_loop" for n in result.nodes)
    
    def test_function_definition(self):
        code = """
def add(a, b):
    return a + b
"""
        result = parse_code(code)
        assert result.success
        assert "add" in result.functions
    
    def test_if_statement(self):
        code = """
x = 10
if x > 5:
    print("big")
else:
    print("small")
"""
        result = parse_code(code)
        assert result.success
        assert any(n.node_type.value == "if_statement" for n in result.nodes)
    
    def test_syntax_error(self):
        code = "def broken("
        result = parse_code(code)
        assert not result.success
        assert result.error is not None
    
    def test_while_loop(self):
        code = """
i = 0
while i < 5:
    i += 1
"""
        result = parse_code(code)
        assert result.success
        assert any(n.node_type.value == "while_loop" for n in result.nodes)
    
    def test_bubble_sort(self):
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
        result = parse_code(code)
        assert result.success
        assert "bubble_sort" in result.functions


class TestStepGenerator:
    """Tests for Step Generator"""
    
    def test_generates_steps(self):
        code = """
x = 5
y = 10
z = x + y
"""
        parse_result = parse_code(code)
        steps = generate_steps(parse_result)
        assert len(steps) >= 3
    
    def test_step_has_required_fields(self):
        code = "x = 5"
        parse_result = parse_code(code)
        steps = generate_steps(parse_result)
        
        assert len(steps) > 0
        step = steps[0]
        
        assert "step_id" in step
        assert "step_type" in step
        assert "line_number" in step
        assert "code_snippet" in step
        assert "state_before" in step
        assert "state_after" in step
        assert "visualization" in step
    
    def test_loop_generates_iterations(self):
        code = """
for i in range(3):
    x = i
"""
        parse_result = parse_code(code)
        steps = generate_steps(parse_result)
        
        iteration_steps = [s for s in steps if s["step_type"] == "loop_iteration"]
        assert len(iteration_steps) >= 3
    
    def test_with_input_values(self):
        code = """
arr = [1, 2, 3]
total = sum(arr)
"""
        parse_result = parse_code(code)
        steps = generate_steps(parse_result, {"arr": [1, 2, 3]})
        
        # Should have initialization step
        init_steps = [s for s in steps if s["step_type"] == "initialize"]
        assert len(init_steps) >= 1


class TestAIEngine:
    """Tests for AI Explanation Engine"""
    
    def test_create_engine(self):
        engine = create_ai_engine(style="beginner")
        assert engine is not None
    
    def test_explain_step(self):
        engine = create_ai_engine()
        
        step = {
            "step_id": 1,
            "step_type": "assign",
            "line_number": 1,
            "code_snippet": "x = 5",
            "description": "Assign 5 to x",
            "state_before": {},
            "state_after": {"x": 5},
            "visualization": {},
            "ai_context": {
                "action": "assignment",
                "targets": ["x"],
                "value": "5"
            }
        }
        
        explanation = engine.explain_step(step)
        
        assert "summary" in explanation
        assert "detailed_explanation" in explanation
        assert "why_this_happens" in explanation
    
    def test_complexity_analysis(self):
        engine = create_ai_engine()
        
        steps = [
            {"step_type": "start_loop", "ai_context": {}},
            {"step_type": "loop_iteration", "ai_context": {"iteration_number": 1}},
            {"step_type": "loop_iteration", "ai_context": {"iteration_number": 2}},
            {"step_type": "end_loop", "ai_context": {}}
        ]
        
        complexity = engine.get_complexity_analysis(steps)
        
        assert "time_complexity" in complexity
        assert "space_complexity" in complexity
        assert "explanation" in complexity
    
    def test_answer_question(self):
        engine = create_ai_engine()
        
        answer = engine.answer_question(
            question="What is the time complexity?",
            code="for i in range(n): print(i)",
            steps=[]
        )
        
        assert answer is not None
        assert len(answer) > 0


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_pipeline_bubble_sort(self):
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

result = bubble_sort([64, 34, 25])
"""
        # Parse
        parse_result = parse_code(code)
        assert parse_result.success
        
        # Generate steps
        steps = generate_steps(parse_result, {"arr": [64, 34, 25]})
        assert len(steps) > 0
        
        # Generate explanations
        engine = create_ai_engine()
        explanations = engine.explain_all_steps(steps)
        assert len(explanations) == len(steps)
        
        # Get complexity
        complexity = engine.get_complexity_analysis(steps)
        assert complexity["loop_count"] >= 1
    
    def test_full_pipeline_binary_search(self):
        code = """
def binary_search(arr, target):
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
"""
        parse_result = parse_code(code)
        assert parse_result.success
        assert "binary_search" in parse_result.functions


# Run tests
if __name__ == "__main__":
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests from each class
    suite.addTests(loader.loadTestsFromTestCase(type('TestASTParser', (unittest.TestCase,), {
        k: v for k, v in TestASTParser.__dict__.items() if k.startswith('test_')
    })))
    suite.addTests(loader.loadTestsFromTestCase(type('TestStepGenerator', (unittest.TestCase,), {
        k: v for k, v in TestStepGenerator.__dict__.items() if k.startswith('test_')
    })))
    suite.addTests(loader.loadTestsFromTestCase(type('TestAIEngine', (unittest.TestCase,), {
        k: v for k, v in TestAIEngine.__dict__.items() if k.startswith('test_')
    })))
    suite.addTests(loader.loadTestsFromTestCase(type('TestIntegration', (unittest.TestCase,), {
        k: v for k, v in TestIntegration.__dict__.items() if k.startswith('test_')
    })))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
