"""
CodeFlow AI - AST Parser Module
Safely parses Python code using the ast module without execution.
"""

import ast
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    """Types of AST nodes we track for visualization."""
    ASSIGNMENT = "assignment"
    FUNCTION_DEF = "function_def"
    FUNCTION_CALL = "function_call"
    IF_STATEMENT = "if_statement"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    RETURN = "return"
    EXPRESSION = "expression"
    COMPARISON = "comparison"
    BINARY_OP = "binary_op"
    LIST_OP = "list_operation"
    VARIABLE = "variable"
    BREAK = "break"
    CONTINUE = "continue"


@dataclass
class ParsedNode:
    """Represents a parsed AST node with visualization metadata."""
    node_type: NodeType
    line_number: int
    end_line: int
    code_snippet: str
    details: Dict[str, Any] = field(default_factory=dict)
    children: List['ParsedNode'] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing Python code."""
    success: bool
    nodes: List[ParsedNode] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    source_lines: List[str] = field(default_factory=list)


class SafeASTParser:
    """
    Safely parses Python code using AST without executing it.
    Extracts structure for visualization.
    """
    
    # Potentially dangerous nodes to flag (not block, just warn)
    UNSAFE_NODES = {
        'Import', 'ImportFrom', 'Exec', 'Eval',
        'Open', 'Subprocess', 'Os', 'Sys'
    }
    
    def __init__(self):
        self.source_lines: List[str] = []
        self.variables: Dict[str, Any] = {}
        self.functions: List[str] = []
        self.warnings: List[str] = []
    
    def parse(self, code: str) -> ParseResult:
        """
        Parse Python code and return structured representation.
        
        Args:
            code: Python source code string
            
        Returns:
            ParseResult with parsed nodes and metadata
        """
        self.source_lines = code.split('\n')
        self.variables = {}
        self.functions = []
        self.warnings = []
        
        try:
            tree = ast.parse(code)
            nodes = self._process_body(tree.body)
            
            return ParseResult(
                success=True,
                nodes=nodes,
                variables=self.variables,
                functions=self.functions,
                source_lines=self.source_lines
            )
            
        except SyntaxError as e:
            return ParseResult(
                success=False,
                error=f"Syntax Error at line {e.lineno}: {e.msg}",
                source_lines=self.source_lines
            )
        except (ValueError, TypeError, AttributeError) as e:
            return ParseResult(
                success=False,
                error=f"Parse Error: {str(e)}",
                source_lines=self.source_lines
            )
    
    def _get_code_snippet(self, node: ast.AST) -> str:
        """Extract the source code for a given AST node."""
        try:
            start_line = getattr(node, 'lineno', 1) - 1
            end_line = getattr(node, 'end_lineno', getattr(node, 'lineno', 1))
            return '\n'.join(self.source_lines[start_line:end_line])
        except (AttributeError, IndexError, TypeError):
            return ""
    
    def _process_body(self, body: List[ast.stmt]) -> List[ParsedNode]:
        """Process a list of statements."""
        nodes = []
        for stmt in body:
            parsed = self._process_node(stmt)
            if parsed:
                nodes.append(parsed)
        return nodes
    
    def _process_node(self, node: ast.AST) -> Optional[ParsedNode]:
        """Process a single AST node and return ParsedNode."""
        
        if isinstance(node, ast.Assign):
            return self._process_assignment(node)
        
        elif isinstance(node, ast.AugAssign):
            return self._process_aug_assignment(node)
        
        elif isinstance(node, ast.FunctionDef):
            return self._process_function_def(node)
        
        elif isinstance(node, ast.If):
            return self._process_if(node)
        
        elif isinstance(node, ast.For):
            return self._process_for(node)
        
        elif isinstance(node, ast.While):
            return self._process_while(node)
        
        elif isinstance(node, ast.Return):
            return self._process_return(node)
        
        elif isinstance(node, ast.Expr):
            return self._process_expression(node)
        
        elif isinstance(node, ast.Break):
            return ParsedNode(
                node_type=NodeType.BREAK,
                line_number=node.lineno,
                end_line=node.lineno,
                code_snippet="break",
                details={"action": "exit_loop"}
            )
        
        elif isinstance(node, ast.Continue):
            return ParsedNode(
                node_type=NodeType.CONTINUE,
                line_number=node.lineno,
                end_line=node.lineno,
                code_snippet="continue",
                details={"action": "next_iteration"}
            )
        
        return None
    
    def _process_assignment(self, node: ast.Assign) -> ParsedNode:
        """Process variable assignment."""
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
                self.variables[target.id] = self._get_value_repr(node.value)
            elif isinstance(target, ast.Subscript):
                targets.append(self._get_subscript_repr(target))
        
        return ParsedNode(
            node_type=NodeType.ASSIGNMENT,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "targets": targets,
                "value": self._get_value_repr(node.value),
                "value_type": self._get_value_type(node.value)
            }
        )
    
    def _process_aug_assignment(self, node: ast.AugAssign) -> ParsedNode:
        """Process augmented assignment (+=, -=, etc.)."""
        target = ""
        if isinstance(node.target, ast.Name):
            target = node.target.id
        
        op_map = {
            ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=",
            ast.Div: "/=", ast.Mod: "%=", ast.Pow: "**="
        }
        op = op_map.get(type(node.op), "?=")
        
        return ParsedNode(
            node_type=NodeType.ASSIGNMENT,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "targets": [target],
                "operator": op,
                "value": self._get_value_repr(node.value),
                "augmented": True
            }
        )
    
    def _process_function_def(self, node: ast.FunctionDef) -> ParsedNode:
        """Process function definition."""
        self.functions.append(node.name)
        
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        return ParsedNode(
            node_type=NodeType.FUNCTION_DEF,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "name": node.name,
                "arguments": args,
                "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
            },
            children=self._process_body(node.body)
        )
    
    def _process_if(self, node: ast.If) -> ParsedNode:
        """Process if statement."""
        condition_details = self._analyze_condition(node.test)
        
        children = self._process_body(node.body)
        
        # Process elif/else
        else_children: List[ParsedNode] = []
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # It's an elif
                elif_node = self._process_node(node.orelse[0])
                if elif_node is not None:
                    else_children = [elif_node]
            else:
                # It's an else block
                else_children = self._process_body(node.orelse)
        
        # Combine children, filtering out None values
        all_children: List[ParsedNode] = children + else_children
        
        return ParsedNode(
            node_type=NodeType.IF_STATEMENT,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "condition": self._get_condition_repr(node.test),
                "condition_analysis": condition_details,
                "has_else": bool(node.orelse)
            },
            children=all_children
        )
    
    def _process_for(self, node: ast.For) -> ParsedNode:
        """Process for loop."""
        target = ""
        if isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node.target, ast.Tuple):
            target = ", ".join(
                n.id for n in node.target.elts if isinstance(n, ast.Name)
            )
        
        iterable_info = self._analyze_iterable(node.iter)
        
        return ParsedNode(
            node_type=NodeType.FOR_LOOP,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "loop_variable": target,
                "iterable": iterable_info,
                "iteration_type": self._get_iteration_type(node.iter)
            },
            children=self._process_body(node.body)
        )
    
    def _process_while(self, node: ast.While) -> ParsedNode:
        """Process while loop."""
        condition_details = self._analyze_condition(node.test)
        
        return ParsedNode(
            node_type=NodeType.WHILE_LOOP,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "condition": self._get_condition_repr(node.test),
                "condition_analysis": condition_details
            },
            children=self._process_body(node.body)
        )
    
    def _process_return(self, node: ast.Return) -> ParsedNode:
        """Process return statement."""
        return ParsedNode(
            node_type=NodeType.RETURN,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            code_snippet=self._get_code_snippet(node),
            details={
                "value": self._get_value_repr(node.value) if node.value else None
            }
        )
    
    def _process_expression(self, node: ast.Expr) -> Optional[ParsedNode]:
        """Process expression statement."""
        if isinstance(node.value, ast.Call):
            return self._process_call(node.value, node.lineno)
        return None
    
    def _process_call(self, node: ast.Call, line_no: int) -> ParsedNode:
        """Process function call."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = f"{self._get_value_repr(node.func.value)}.{node.func.attr}"
        
        args = [self._get_value_repr(arg) for arg in node.args]
        
        return ParsedNode(
            node_type=NodeType.FUNCTION_CALL,
            line_number=line_no,
            end_line=getattr(node, 'end_lineno', line_no),
            code_snippet=self._get_code_snippet(node),
            details={
                "function": func_name,
                "arguments": args,
                "is_builtin": func_name in ['print', 'len', 'range', 'enumerate', 'zip', 'sorted', 'reversed']
            }
        )
    
    def _get_value_repr(self, node: Optional[ast.AST]) -> str:
        """Get string representation of a value node."""
        if node is None:
            return "None"
        
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            elements = [self._get_value_repr(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ast.Dict):
            pairs = []
            for k, v in zip(node.keys, node.values):
                pairs.append(f"{self._get_value_repr(k)}: {self._get_value_repr(v)}")
            return "{" + ", ".join(pairs) + "}"
        elif isinstance(node, ast.BinOp):
            left = self._get_value_repr(node.left)
            right = self._get_value_repr(node.right)
            op = self._get_binop_symbol(node.op)
            return f"{left} {op} {right}"
        elif isinstance(node, ast.Compare):
            return self._get_condition_repr(node)
        elif isinstance(node, ast.Call):
            func = self._get_value_repr(node.func)
            args = [self._get_value_repr(a) for a in node.args]
            return f"{func}({', '.join(args)})"
        elif isinstance(node, ast.Subscript):
            return self._get_subscript_repr(node)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_value_repr(node.value)}.{node.attr}"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_value_repr(e) for e in node.elts]
            return f"({', '.join(elements)})"
        
        return "<?>"
    
    def _get_subscript_repr(self, node: ast.Subscript) -> str:
        """Get string representation of subscript access."""
        value = self._get_value_repr(node.value)
        slice_repr = self._get_value_repr(node.slice)
        return f"{value}[{slice_repr}]"
    
    def _get_value_type(self, node: ast.AST) -> str:
        """Determine the type of a value node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            return "call_result"
        return "unknown"
    
    def _get_binop_symbol(self, op: ast.operator) -> str:
        """Get symbol for binary operator."""
        op_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.Mod: "%", ast.Pow: "**", ast.FloorDiv: "//",
            ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
            ast.BitXor: "^", ast.BitAnd: "&"
        }
        return op_map.get(type(op), "?")
    
    def _get_condition_repr(self, node: ast.AST) -> str:
        """Get string representation of a condition."""
        if isinstance(node, ast.Compare):
            left = self._get_value_repr(node.left)
            parts = [left]
            for op, comp in zip(node.ops, node.comparators):
                op_symbol = self._get_cmpop_symbol(op)
                parts.append(op_symbol)
                parts.append(self._get_value_repr(comp))
            return " ".join(parts)
        elif isinstance(node, ast.BoolOp):
            op = " and " if isinstance(node.op, ast.And) else " or "
            values = [self._get_condition_repr(v) for v in node.values]
            return op.join(values)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"not {self._get_condition_repr(node.operand)}"
        else:
            return self._get_value_repr(node)
    
    def _get_cmpop_symbol(self, op: ast.cmpop) -> str:
        """Get symbol for comparison operator."""
        op_map = {
            ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
            ast.In: "in", ast.NotIn: "not in"
        }
        return op_map.get(type(op), "?")
    
    def _analyze_condition(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze a condition for AI explanation."""
        analysis = {
            "type": "simple",
            "components": [],
            "operators": []
        }
        
        if isinstance(node, ast.Compare):
            analysis["type"] = "comparison"
            analysis["left"] = self._get_value_repr(node.left)
            analysis["operators"] = [self._get_cmpop_symbol(op) for op in node.ops]
            analysis["comparators"] = [self._get_value_repr(c) for c in node.comparators]
        elif isinstance(node, ast.BoolOp):
            analysis["type"] = "boolean"
            analysis["operator"] = "and" if isinstance(node.op, ast.And) else "or"
            analysis["components"] = [self._analyze_condition(v) for v in node.values]
        
        return analysis
    
    def _analyze_iterable(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze an iterable for visualization."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "range":
                    args = [self._get_value_repr(a) for a in node.args]
                    return {
                        "type": "range",
                        "args": args,
                        "description": f"range({', '.join(args)})"
                    }
                elif node.func.id == "enumerate":
                    return {
                        "type": "enumerate",
                        "target": self._get_value_repr(node.args[0]) if node.args else "?",
                        "description": "enumerate with index"
                    }
        elif isinstance(node, ast.Name):
            return {
                "type": "variable",
                "name": node.id,
                "description": f"iterating over {node.id}"
            }
        elif isinstance(node, ast.List):
            return {
                "type": "list_literal",
                "elements": [self._get_value_repr(e) for e in node.elts],
                "description": "list literal"
            }
        
        return {"type": "unknown", "description": self._get_value_repr(node)}
    
    def _get_iteration_type(self, node: ast.AST) -> str:
        """Determine the type of iteration."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "range":
                return "index_based"
            elif node.func.id == "enumerate":
                return "indexed_iteration"
        return "element_based"


def parse_code(code: str) -> ParseResult:
    """Convenience function to parse code."""
    parser = SafeASTParser()
    return parser.parse(code)
