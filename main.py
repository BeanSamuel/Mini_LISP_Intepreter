import re
import os
from copy import deepcopy


class EnvironmentStack:
    """
    用來管理多層變數作用域的堆疊 (Stack)。
    最底層 (index=0) 為全域作用域。
    之後每 push_scope() 一次，就新增一層作用域。
    pop_scope() 後便回到前一層作用域。
    """
    def __init__(self):
        # 預設只有一層 (全域)
        self.scopes = [{}]

    def push_scope(self, new_scope: dict = None) -> None:
        self.scopes.append(new_scope if new_scope else {})

    def pop_scope(self) -> None:
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise RuntimeError("Cannot pop the global scope.")
        
    def define(self, key: str, value, allow_override: bool = False) -> None:
        """
        在『最上層作用域』定義/綁定一個變數。
        allow_override = False 時，若該變數在「整個 stack」中已存在則拋錯。
        如果希望只檢查當前作用域，可自行調整。
        """
        if not allow_override:
            for scope in self.scopes:
                if key in scope:
                    raise NameError(f"Variable '{key}' already exists.")
        self.scopes[-1][key] = value

    def get_value(self, key: str):
        """
        從最上層往下尋找變數定義，找到就回傳，否則拋出 NameError。
        """
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        raise NameError(f"Variable '{key}' does not exist.")

    def __repr__(self):
        return f"Top scope: {self.scopes[-1]}"

class ASTNode:
    """
    表示程式中的基本單元 (token or expression)，例如：
    - unit_type = 'var': 變數
    - unit_type = 'num': 整數
    - unit_type = 'bool': 布林值
    - unit_type = 'keyword': 關鍵字 (如 'define', 'fun' 等)
    - unit_type = 'expr': 以括號包裹的運算式
    - unit_type = 'fun': 自訂函數
    - unit_type = 'others': 暫時無法辨識的符號或型別
    name 則是此單元的具體內容。對於 'expr' 來說，name 即括號內部的字串。
    """

    def __init__(self, unit_type: str, name):
        self.unit_type = unit_type
        self.name = name

    def __repr__(self):
        return f"{self.unit_type} {self.name}"

    def get_value(self):
        """
        依照自身的 unit_type 來決定「求值」行為。
        - var: 從 vars_manager (EnvironmentStack) 中取值
        - expr: 進一步解析成子 AST，並執行
        - 其餘 (num, bool, fun, etc.): 直接回傳自己
        """
        if self.unit_type == 'var':
            return vars_manager.get_value(self.name)
        elif self.unit_type == 'expr':
            return self._evaluate_expression()
        else:
            return self

    def _evaluate_expression(self):
        units = parse_tokens(self.name)
        if not units:
            raise SyntaxError(f"syntax error")

        # 第一個單元應該是運算子或函式
        first_unit = units[0].get_value()
        op_name, op_type = first_unit.name, first_unit.unit_type
        args = units[1:]

        # 如果第一個單元是一個 fun (自訂函數)
        if op_type == 'fun':
            func_obj = op_name
            return func_obj(args)

        # 如果第一個單元是關鍵字或符號
        if op_name == 'define':
            return self._process_define(args)
        elif op_name in ['+', '-', '*', '/', 'mod', '>', '<', 'and', 'or', 'not', '=']:
            return self._process_operator(op_name, args)
        elif op_name in ['print-num', 'print-bool']:
            return self._process_print(op_name, args)
        elif op_name == 'fun':
            return self._process_fun(args)
        elif op_name == 'if':
            return self._process_if(args)

        raise ValueError(f"Invalid operator: {op_name}")

    def _process_define(self, args: list):
        if len(args) != 2:
            raise SyntaxError(f"syntax error")
        
        var_unit = args[0]
        if var_unit.unit_type != 'var':
            raise SyntaxError(f"syntax error")

        value_unit = args[1].get_value()
        vars_manager.define(var_unit.name, value_unit)
        return ASTNode('keyword', 'define')

    def _process_operator(self, op_name: str, args: list):
        # 基本運算子長度檢查
        if op_name in ['+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or']:
            if len(args) < 2:
                raise SyntaxError("syntax error")
        elif op_name == 'not':
            if len(args) != 1:
                raise SyntaxError("syntax error")

        # 預期型別
        expected_type = 'bool' if op_name in ['and', 'or', 'not'] else 'num'
        
        result = None
        for idx, arg in enumerate(args):
            evaluated_arg = arg.get_value()

            # [新增判斷] 如果是 'others'，則直接回傳 syntax error
            if evaluated_arg.unit_type == 'others':
                raise SyntaxError("syntax error")

            # 若不是 'others'，但也不是預期型別，就回傳 type error
            if evaluated_arg.unit_type != expected_type:
                raise TypeError("Type error!")

            # 處理單一參數的 not
            if op_name == 'not':
                return ASTNode('bool', not evaluated_arg.name)

            # 初始化第一個參數
            if result is None:
                result = evaluated_arg.name
                continue

            # 逐一計算
            if op_name == '+':
                result += evaluated_arg.name
            elif op_name == '-':
                result -= evaluated_arg.name
                return ASTNode('num', result)
            elif op_name == '*':
                result *= evaluated_arg.name
            elif op_name == '/':
                result //= evaluated_arg.name
                return ASTNode('num', result)
            elif op_name == 'mod':
                result %= evaluated_arg.name
                return ASTNode('num', result)
            elif op_name == '>':
                result = (result > evaluated_arg.name)
                return ASTNode('bool', result)
            elif op_name == '<':
                result = (result < evaluated_arg.name)
                return ASTNode('bool', result)
            elif op_name == '=':
                result = (result == evaluated_arg.name)
            elif op_name == 'and':
                result = result and evaluated_arg.name
            elif op_name == 'or':
                result = result or evaluated_arg.name

        # 最後判斷 result 是否變成 bool
        if isinstance(result, bool):
            return ASTNode('bool', result)
        else:
            return ASTNode('num', result)

    def _process_print(self, op_name: str, args: list):
        if len(args) != 1:
            raise SyntaxError(f"syntax error")

        # e.g. 'print-bool' -> ['print', 'bool']
        _, expected_output_type = op_name.split('-')
        evaluated_arg = args[0].get_value()

        if evaluated_arg.unit_type != expected_output_type:
            raise TypeError(f"Type error!")

        if expected_output_type == 'bool':
            print('#t' if evaluated_arg.name else '#f')
        else:
            print(evaluated_arg.name)

        # 回傳被印出的值
        return evaluated_arg

    def _process_fun(self, args: list):
        if len(args) < 2:
            raise SyntaxError(f"syntax error")

        param_list_unit = args[0]
        if param_list_unit.unit_type != 'expr':
            raise SyntaxError(f"syntax error")

        middle_defines = args[1:-1]
        last_expr = args[-1]

        # 中間的 define 全部執行一次
        for mid in middle_defines:
            mid_unit = mid.get_value()
            if mid_unit.unit_type != 'keyword' or mid_unit.name != 'define':
                raise SyntaxError(f"syntax error")

        # 建立一個函式 ASTNode('fun', UserFunction(...))
        return ASTNode('fun', UserFunction(param_list_unit, last_expr))

    def _process_if(self, args: list):
        if len(args) != 3:
            raise SyntaxError(f"syntax error")

        condition = args[0].get_value()
        if condition.unit_type != 'bool':
            raise TypeError("Type error!")

        if condition.name:
            return args[1].get_value()
        else:
            return args[2].get_value()


class UserFunction:
    """
    用來表示使用者自訂函式。
    包含:
    - 參數列表
    - 函式體 (expression_unit)
    - 建立此函式時的環境快照 (capture_vars)
    """
    def __init__(self, param_list_unit: ASTNode, expression_unit: ASTNode):
        params_units = parse_tokens(param_list_unit.name)
        for p in params_units:
            if p.unit_type != 'var':
                raise SyntaxError(f"syntax error")
        self.params_list = [p.name for p in params_units]
        self.expr = expression_unit
        # [CHANGED] 改用 deepcopy(vars_manager.scopes[-1]) 建立快照
        self.capture_vars = deepcopy(vars_manager.scopes[-1])

    def __call__(self, args: list):
        # 檢查參數數量
        if len(args) != len(self.params_list):
            raise SyntaxError(f"syntax error")

        # 先 push_scope()，再把帶入參數 define 進當前作用域
        vars_manager.push_scope(deepcopy(self.capture_vars))

        computed_args = [arg.get_value() for arg in args]
        for param_name, arg_value in zip(self.params_list, computed_args):
            # 這裡使用 allow_override=True 因為要直接賦值給參數
            vars_manager.define(param_name, arg_value, allow_override=True)

        # 執行函式體
        result = self.expr.get_value()

        # 彈出函式作用域
        vars_manager.pop_scope()

        return result
    
    def __repr__(self):
        return f"UserFunction(params={self.params_list}, body={self.expr})"


def parse_parenthesized_expression(source_str: str):
    s = source_str.strip()
    if not s.startswith('('):
        raise SyntaxError(f"syntax error")

    depth = 0
    closing_index = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                raise SyntaxError(f"syntax error")
        if depth == 0:
            closing_index = i
            break
    
    if depth != 0:
        raise SyntaxError(f"syntax error")

    # 取出最外層括號內的字串
    inner_expr = s[1:closing_index].strip()
    # 剩餘的字串
    remaining_str = s[closing_index + 1:].strip()

    return ASTNode('expr', inner_expr), remaining_str


def parse_token(source_str: str):
    tokens = source_str.split(maxsplit=1)
    if len(tokens) == 1:
        token, remaining_str = tokens[0], ''
    else:
        token, remaining_str = tokens[0], tokens[1]

    # 布林值
    if token == '#t':
        return ASTNode('bool', True), remaining_str
    if token == '#f':
        return ASTNode('bool', False), remaining_str
    
    # 關鍵字
    if token in ['print-num', 'print-bool', 'define', 'if', 'and', 'or', 'not', 'then', 'fun', 'else', 'mod']:
        return ASTNode('keyword', token), remaining_str
    
    # 數字
    if re.fullmatch(r'0|([1-9]\d*)|(-[1-9]\d*)', token):
        value = int(token)
        if not(-2**31 <= value <= 2**31 - 1):
            raise ValueError(f"Invalid integer: {value}")
        return ASTNode('num', value), remaining_str

    # 變數名稱
    if re.fullmatch(r'[a-zA-Z][a-zA-Z0-9\-]*', token):
        return ASTNode('var', token), remaining_str
    
    # 其他未知符號
    return ASTNode('others', token), remaining_str


def parse_tokens(source_str: str):
    """
    將字串解析成一串 ASTNode。
    e.g. "x (define y 10) #t" -> [var:x, expr:(define y 10), bool:#t]
    """
    results = []
    s = source_str.strip()

    while s:
        if s.startswith('('):
            expr_node, s = parse_parenthesized_expression(s)
            results.append(expr_node)
        else:
            token_node, s = parse_token(s)
            results.append(token_node)
        s = s.strip()

    return results

def main():
    global vars_manager
    vars_manager = EnvironmentStack()

    test_dir = './test_data/public_test_data'  # Path to the test directory

    for file in sorted(os.listdir(test_dir)):
        if file:
            print(f"==================== 執行檔案: {file} ====================")
            with open(os.path.join(test_dir, file), 'r', encoding='utf-8') as f:
                content = ' '.join(line.strip() for line in f)

            vars_manager = EnvironmentStack()

            try:
                while content:
                    expr_node, content = parse_parenthesized_expression(content.strip())
                    expr_node.get_value()
            except Exception as e:
                print(e)

if __name__ == '__main__':
    main()
