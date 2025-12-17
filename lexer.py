import sys
from enum import Enum
from typing import Optional, Tuple, Dict, List, Any

# типы токенов
class TokenType(Enum):
    KEYWORD = 1
    DELIMITER = 2
    NUMBER = 3
    IDENTIFIER = 4
    EOF = 5
    ERROR = 6


class State(Enum):
    H = 0  # Начало
    I = 1  # Идентификатор
    N2 = 2  # Двоичное число
    N8 = 3  # Восьмеричное число
    N10 = 4  # Десятичное число
    N16 = 5  # Шестнадцатеричное число
    B = 6  # 'B'
    O = 7  # 'O'
    D = 8  # 'D'
    HX = 9  # 'H'
    E11 = 10  # 'E' после цифр
    E12 = 11  # Цифры после E без знака
    E13 = 12  # Цифры после знака порядка
    ZN = 13  # Знак порядка (+/-)
    P1 = 14  # Точка
    P2 = 15  # Дробная часть
    M1 = 16  # '<'
    M2 = 17  # '>'
    C1 = 18  # Комментарий: {
    C2 = 19  # Комментарий: внутри
    C3 = 20  # Комментарий: }
    OG = 21  # Ограничитель
    V = 22  # Выход (токен готов)
    ER = 23  # Ошибка


class Token:
    def __init__(self, type: TokenType, value: str, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
        self.num_value: Optional[float] = None
        self.num_type: Optional[str] = None  # '%', '#', '$'

    def __repr__(self):
        if self.num_value is not None:
            return f"Token({self.type}, '{self.value}', line={self.line}, col={self.col}, num={self.num_value}, num_type={self.num_type})"
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.col})"


class Lexer:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.tokens: List[Token] = []
        self.identifiers: Dict[str, int] = {}
        self.numbers: List[Dict[str, Any]] = []
        self.current_char: Optional[str] = None
        self.line = 1
        self.col = 0
        self.s = ""  # буфер
        self.t: Optional[TokenType] = None
        self.z = 0
        self.num_value: Optional[float] = None
        self.num_type: Optional[str] = None
        self.file = open(input_file, 'r', encoding='utf-8')
        self.read_char()

        # Таблицы для варианта 1
        self.KEYWORDS = {
            'read': 1, 'write': 2, 'if': 3, 'then': 4, 'else': 5,
            'for': 6, 'to': 7, 'while': 8, 'do': 9, 'true': 10,
            'false': 11, 'or': 12, 'and': 13, 'not': 14, 'as': 15
        }
        self.DELIMITERS = {
            '{': 1, '}': 2, '%': 3, '#': 4, '$': 5, ',': 6, ';': 7,
            '[': 8, ']': 9, ':': 10, '(': 11, ')': 12, '+': 13,
            '-': 14, '*': 15, '/': 16, '=': 17, '<>': 18, '<': 19,
            '>': 20, '<=': 21, '>=': 22
        }

    def read_char(self):
        self.current_char = self.file.read(1)
        if self.current_char:
            self.col += 1
            if self.current_char == '\n':
                self.line += 1
                self.col = 0
        else:
            self.current_char = None

    def is_letter(self, ch: str) -> bool:
        return ch is not None and (('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ch == '_')

    def is_digit(self, ch: str) -> bool:
        return ch is not None and '0' <= ch <= '9'

    def is_bin_digit(self, ch: str) -> bool:
        return ch is not None and ch in '01'

    def is_oct_digit(self, ch: str) -> bool:
        return ch is not None and '0' <= ch <= '7'

    def is_hex_digit(self, ch: str) -> bool:
        return ch is not None and (('0' <= ch <= '9') or ('a' <= ch <= 'f') or ('A' <= ch <= 'F'))

    def clear_buffer(self):
        self.s = ""

    def add_to_buffer(self, ch: str):
        self.s += ch

    def look(self, table: str) -> int:
        if table == 'keywords':
            return self.KEYWORDS.get(self.s, 0)
        elif table == 'delimiters':
            return self.DELIMITERS.get(self.s, 0)
        elif table == 'identifiers':
            return self.identifiers.get(self.s, 0)
        return 0

    def put_identifier(self) -> int:
        # Если идентификатор уже есть, возвращаем его номер
        if self.s in self.identifiers:
            return self.identifiers[self.s]

        # Иначе добавляем новый
        idx = len(self.identifiers) + 1
        self.identifiers[self.s] = idx
        return idx

    def put_number(self, value: float, num_type: str) -> int:
        # Проверяем, есть ли уже такое число в таблице
        for i, num in enumerate(self.numbers):
            if num['string'] == self.s and num['type'] == num_type:
                return i + 1  # Возвращаем существующий номер

        # Если нет, добавляем новое
        idx = len(self.numbers) + 1
        self.numbers.append({
            'string': self.s,
            'value': value,
            'type': num_type
        })
        return idx

    def translate(self, base: int) -> float:
        try:
            # Убираем суффикс (B, O, H и т.д.)
            num_str = self.s
            if num_str[-1].upper() in ('B', 'O', 'D', 'H'):
                num_str = num_str[:-1]
            return int(num_str, base)
        except ValueError:
            return 0.0

    def convert_real(self) -> float:
        try:
            return float(self.s)
        except ValueError:
            return 0.0

    def scan(self):
        current_state = State.H
        self.clear_buffer()
        self.t = None
        self.z = 0
        self.num_value = None
        self.num_type = None

        while True:
            ch = self.current_char

            if ch is None and current_state == State.H:
                self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
                break

            if current_state == State.H:
                while ch in (' ', '\n', '\t', '\r'):
                    self.read_char()
                    ch = self.current_char

                if ch is None:
                    self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
                    break
                elif self.is_letter(ch):
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.I
                elif self.is_digit(ch):
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.N10
                elif ch == '.':
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.P1
                elif ch == '{':
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS[ch]
                    self.read_char()
                    current_state = State.V
                elif ch == '<':
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.M1
                elif ch == '>':
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.M2
                elif ch == '}':
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS[ch]
                    self.read_char()
                    current_state = State.V
                else:
                    self.clear_buffer()
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.OG

            elif current_state == State.I:
                while self.is_letter(ch) or self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    ch = self.current_char

                if self.s in self.KEYWORDS:
                    self.t = TokenType.KEYWORD
                    self.z = self.KEYWORDS[self.s]
                else:
                    self.t = TokenType.IDENTIFIER
                    self.z = self.put_identifier()
                current_state = State.V

            elif current_state == State.N10:
                while self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    ch = self.current_char

                if ch == '.':
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.P1
                elif ch in ('E', 'e'):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.E11
                elif ch in ('B', 'b'): # 2
                    if all(c in '01' for c in self.s):
                        self.add_to_buffer(ch)
                        self.read_char()
                        current_state = State.B
                    else:
                        current_state = State.ER
                elif ch in ('O', 'o'):# 8
                    if all('0' <= c <= '7' for c in self.s):
                        self.read_char()
                        current_state = State.O
                    else:
                        current_state = State.ER
                elif ch in ('H', 'h'): #16
                    self.read_char()
                    current_state = State.HX
                elif ch in ('D', 'd'): #10
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.D
                else: #10
                    self.num_value = float(self.s)
                    self.num_type = '%'
                    self.t = TokenType.NUMBER
                    self.z = self.put_number(self.num_value, self.num_type)
                    current_state = State.V

            elif current_state == State.B:
                num_str = self.s[:-1]
                self.num_value = int(num_str, 2)
                self.num_type = '%'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.O:
                self.num_value = int(self.s, 8)
                self.num_type = '%'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.D:
                num_str = self.s[:-1]
                self.num_value = float(num_str)
                self.num_type = '%' if '.' not in num_str else '#'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.HX:
                self.num_value = int(self.s, 16)
                self.num_type = '%'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.P1:
                if self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.P2
                else:
                    current_state = State.ER

            elif current_state == State.P2:
                while self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    ch = self.current_char

                if ch in ('E', 'e'):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.E11
                else:
                    self.num_value = float(self.s)
                    self.num_type = '#'
                    self.t = TokenType.NUMBER
                    self.z = self.put_number(self.num_value, self.num_type)
                    current_state = State.V

            elif current_state == State.E11:
                if ch in ('+', '-'):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.ZN
                elif self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.E12
                else:
                    current_state = State.ER

            elif current_state == State.ZN:
                if self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    current_state = State.E13
                else:
                    current_state = State.ER

            elif current_state == State.E12:
                while self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    ch = self.current_char

                self.num_value = float(self.s)
                self.num_type = '#'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.E13:
                while self.is_digit(ch):
                    self.add_to_buffer(ch)
                    self.read_char()
                    ch = self.current_char

                self.num_value = float(self.s)
                self.num_type = '#'
                self.t = TokenType.NUMBER
                self.z = self.put_number(self.num_value, self.num_type)
                current_state = State.V

            elif current_state == State.M1:
                if ch == '>':
                    self.add_to_buffer(ch)
                    self.read_char()
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS.get(self.s, 0)
                    current_state = State.V
                elif ch == '=':
                    self.add_to_buffer(ch)
                    self.read_char()
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS.get(self.s, 0)
                    current_state = State.V
                else:
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS.get(self.s, 0)
                    current_state = State.V

            elif current_state == State.M2:
                if ch == '=':
                    self.add_to_buffer(ch)
                    self.read_char()
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS.get(self.s, 0)
                    current_state = State.V
                else:
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS.get(self.s, 0)
                    current_state = State.V

            elif current_state == State.OG:
                if self.s in self.DELIMITERS:
                    self.t = TokenType.DELIMITER
                    self.z = self.DELIMITERS[self.s]
                    current_state = State.V
                else:
                    current_state = State.ER

            elif current_state == State.ER:
                self.t = TokenType.ERROR
                current_state = State.V

            if current_state == State.V:
                if self.t == TokenType.ERROR:
                    self.tokens.append(Token(TokenType.ERROR, self.s, self.line, self.col))
                else:
                    token = Token(self.t, self.s, self.line, self.col)
                    if self.num_value is not None:
                        token.num_value = self.num_value
                        token.num_type = self.num_type
                    self.tokens.append(token)

                self.clear_buffer()
                self.t = None
                self.z = 0
                self.num_value = None
                self.num_type = None
                current_state = State.H
    def print_tables(self):
        print("\n" + "=" * 60)
        print("ТАБЛИЦЫ ЛЕКСЕМ")

        print("\n1. Ключевые слова:")
        for kw, idx in sorted(self.KEYWORDS.items(), key=lambda x: x[1]):
            print(f"  {idx:3}. {kw}")

        print("\n2. Разделители:")
        for delim, idx in sorted(self.DELIMITERS.items(), key=lambda x: x[1]):
            print(f"  {idx:3}. '{delim}'")

        print("\n3. Идентификаторы (найдены в программе):")
        if self.identifiers:
            for ident, idx in sorted(self.identifiers.items(), key=lambda x: x[1]):
                print(f"  {idx:3}. {ident}")
        else:
            print("  (нет идентификаторов)")

        print("\n4. Числовые константы:")
        if self.numbers:
            for i, num in enumerate(self.numbers, 1):
                print(f"  {i:3}. {num['string']} = {num['value']} ({num['type']})")
        else:
            print("  (нет числовых констант)")

        print("\n5. Поток токенов:")
        for i, token in enumerate(self.tokens):
            if token.type == TokenType.IDENTIFIER:
                ident_id = self.identifiers.get(token.value, 0)
                print(f"  {i + 1:3}. ({token.type.value},{ident_id}) # '{token.value}'")
            elif token.type == TokenType.NUMBER:
                num_id = next((j + 1 for j, n in enumerate(self.numbers) if n['string'] == token.value), 0)
                print(f"  {i + 1:3}. ({token.type.value},{num_id}) # '{token.value}'")
            elif token.type == TokenType.KEYWORD:
                kw_id = self.KEYWORDS.get(token.value, 0)
                print(f"  {i + 1:3}. ({token.type.value},{kw_id}) # '{token.value}'")
            elif token.type == TokenType.DELIMITER:
                delim_id = self.DELIMITERS.get(token.value, 0)
                print(f"  {i + 1:3}. ({token.type.value},{delim_id}) # '{token.value}'")
            elif token.type == TokenType.EOF:
                print(f"  {i + 1:3}. EOF")
            else:
                print(f"  {i + 1:3}. {token}")

    def print_operation_table(self):
        print("\n" + "=" * 60)
        print("ТАБЛИЦА ДВУМЕСТНЫХ ОПЕРАЦИЙ")
        print(f"{'Операция':<10} {'Тип_1':<10} {'Тип_2':<10} {'Результат':<10}")
        ops = [
            ('+', '%', '%', '%'),
            ('-', '%', '%', '%'),
            ('*', '%', '%', '%'),
            ('/', '%', '%', '%'),
            ('+', '#', '#', '#'),
            ('-', '#', '#', '#'),
            ('*', '#', '#', '#'),
            ('/', '#', '#', '#'),
            ('or', '$', '$', '$'),
            ('and', '$', '$', '$'),
            ('=', '%', '%', '$'),
            ('=', '#', '#', '$'),
            ('=', '$', '$', '$'),
            ('<>', '%', '%', '$'),
            ('<>', '#', '#', '$'),
            ('<>', '$', '$', '$'),
            ('<', '%', '%', '$'),
            ('<', '#', '#', '$'),
            ('<=', '%', '%', '$'),
            ('<=', '#', '#', '$'),
            ('>', '%', '%', '$'),
            ('>', '#', '#', '$'),
            ('>=', '%', '%', '$'),
            ('>=', '#', '#', '$'),
        ]
        for op, t1, t2, res in ops:
            print(f"{op:<10} {t1:<10} {t2:<10} {res:<10}")

# -----------------------------------------------------------
# СИНТАКСИЧЕСКИЙ АНАЛИЗАТОР (рекурсивный спуск, вариант 1)
# -----------------------------------------------------------

class SyntaxError(Exception):
    def __init__(self, message: str, line: int, col: int):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"Syntax error at {line}:{col}: {message}")

class SemanticError(Exception):
    def __init__(self, message: str, line: int, col: int):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"Семантическая ошибка в {line}:{col}: {message}")

class SyntaxAnalyzer:
    def __init__(self, tokens: list, identifiers: Dict[str, int]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else Token(TokenType.EOF, '', 0, 0)
        self.declared_vars: Dict[str, bool] = {}  # имя → описано (True)
        self._last_id_list: List[str] = []  # чтобы передавать имена из parse_id_list
    def advance(self):
        """Переход к следующему токену"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token(TokenType.EOF, '', 0, 0)

    def expect(self, token_type: TokenType, value: str = None, msg: str = "Unexpected token"):
        """Проверка и продвижение"""
        if self.current_token.type != token_type:
            raise SyntaxError(msg, self.current_token.line, self.current_token.col)
        if value is not None and self.current_token.value != value:
            raise SyntaxError(f"Expected '{value}'", self.current_token.line, self.current_token.col)
        self.advance()

    def is_keyword(self, kw: str) -> bool:
        return self.current_token.type == TokenType.KEYWORD and self.current_token.value == kw

    def is_delim(self, d: str) -> bool:
        return self.current_token.type == TokenType.DELIMITER and self.current_token.value == d

    def is_identifier(self) -> bool:
        return self.current_token.type == TokenType.IDENTIFIER

    def is_number(self) -> bool:
        return self.current_token.type == TokenType.NUMBER

    def is_relation_op(self) -> bool:
        return self.current_token.type == TokenType.DELIMITER and self.current_token.value in ('<', '<=', '>', '>=', '=', '<>')

    def is_add_op(self) -> bool:
        return (self.current_token.type == TokenType.DELIMITER and self.current_token.value in ('+', '-')) or \
               (self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'or')

    def is_mult_op(self) -> bool:
        return (self.current_token.type == TokenType.DELIMITER and self.current_token.value in ('*', '/')) or \
               (self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'and')

    # ------------------- Правила грамматики -------------------

    def parse_program(self):
        self.expect(TokenType.DELIMITER, '{')
        self.parse_body()
        self.expect(TokenType.DELIMITER, '}', "Expected '}' at end of program")

    def parse_body(self):
        # Первый элемент: описание или оператор
        if self._is_description_start() or self._is_operator_start():
            self.parse_statement()
            # Далее — ноль или более: ; (описание | оператор)
            while self.is_delim(';'):
                self.advance()
                if self._is_description_start() or self._is_operator_start():
                    self.parse_statement()
                else:
                    break  # допускаем завершение без trailing ;

    def _is_description_start(self) -> bool:
        return self.current_token.type == TokenType.DELIMITER and self.current_token.value in ('%', '#', '$')

    def _is_operator_start(self) -> bool:
        return (self.is_identifier() or
                self.is_keyword('if') or
                self.is_keyword('while') or
                self.is_keyword('for') or
                self.is_keyword('read') or
                self.is_keyword('write') or
                self.is_delim('['))

    def parse_statement(self):
        if self._is_description_start():
            self.parse_description()
        elif self._is_operator_start():
            self.parse_operator()
        else:
            raise SyntaxError("Expected declaration or operator", self.current_token.line, self.current_token.col)

    def parse_description(self):
        if not self._is_description_start():
            raise SyntaxError("Expected type (%/#/$)", self.current_token.line, self.current_token.col)
        self.advance()  # пропускаем тип
        self.parse_id_list()
        # Семантика: запоминаем, что эти переменные описаны
        for name in self._last_id_list:
            if name in self.declared_vars:
                raise SemanticError(f"Повторное описание переменной '{name}'", self.current_token.line,
                                    self.current_token.col)
            self.declared_vars[name] = True

    def parse_id_list(self):
        self._last_id_list = []  # очищаем список
        self._last_id_list.append(self.current_token.value)
        self.expect(TokenType.IDENTIFIER, None, "Expected identifier")
        while self.is_delim(','):
            self.advance()
            self._last_id_list.append(self.current_token.value)
            self.expect(TokenType.IDENTIFIER, None, "Expected identifier after ','")

    def parse_operator(self):
        if self.is_delim('['):
            self.parse_compound()
        elif self.is_keyword('if'):
            self.parse_if()
        elif self.is_keyword('while'):
            self.parse_while()
        elif self.is_keyword('for'):
            self.parse_for()
        elif self.is_keyword('read'):
            self.parse_read()
        elif self.is_keyword('write'):
            self.parse_write()
        elif self.is_identifier():
            self.parse_assignment()
        else:
            raise SyntaxError("Unknown operator", self.current_token.line, self.current_token.col)

    def parse_compound(self):
        self.expect(TokenType.DELIMITER, '[')
        self.parse_operator()
        while self.is_delim(':'):
            self.advance()
            self.parse_operator()
        self.expect(TokenType.DELIMITER, ']')

    def parse_assignment(self):
        var_name = self.current_token.value
        self._check_declared(var_name)
        self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.KEYWORD, 'as')
        self.parse_expression()

    def parse_if(self):
        self.expect(TokenType.KEYWORD, 'if')
        self.parse_expression()
        self.expect(TokenType.KEYWORD, 'then')
        self.parse_operator()
        if self.is_keyword('else'):
            self.advance()
            self.parse_operator()

    def parse_while(self):
        self.expect(TokenType.KEYWORD, 'while')
        self.parse_expression()
        self.expect(TokenType.KEYWORD, 'do')
        self.parse_operator()

    def parse_for(self):
        self.expect(TokenType.KEYWORD, 'for')
        self.parse_assignment()  # <присваивания>
        self.expect(TokenType.KEYWORD, 'to')
        self.parse_expression()
        self.expect(TokenType.KEYWORD, 'do')
        self.parse_operator()

    def parse_read(self):
        self.expect(TokenType.KEYWORD, 'read')
        self.expect(TokenType.DELIMITER, '(')
        self.parse_id_list()
        # Проверка: все переменные в read должны быть описаны
        for name in self._last_id_list:
            if name not in self.declared_vars:
                raise SemanticError(f"Неописанная переменная '{name}' в read", self.current_token.line,
                                    self.current_token.col)
        self.expect(TokenType.DELIMITER, ')')

    def parse_write(self):
        self.expect(TokenType.KEYWORD, 'write')
        self.expect(TokenType.DELIMITER, '(')
        self.parse_expression()
        while self.is_delim(','):
            self.advance()
            self.parse_expression()
        self.expect(TokenType.DELIMITER, ')')

    def parse_expression(self):
        self.parse_sum()
        if self.is_relation_op():
            self.advance()
            self.parse_sum()

    def parse_sum(self):
        self.parse_product()
        while self.is_add_op():
            self.advance()
            self.parse_product()

    def parse_product(self):
        self.parse_factor()
        while self.is_mult_op():
            self.advance()
            self.parse_factor()

    def parse_factor(self):
        if self.is_identifier():
            self._check_declared(self.current_token.value)
            self.advance()
        elif self.is_number():
            self.advance()
        elif self.current_token.value in ('true', 'false'):
            self.advance()
        elif self.is_keyword('not'):
            self.advance()
            self.parse_factor()
        elif self.is_delim('('):
            self.advance()
            self.parse_expression()
            self.expect(TokenType.DELIMITER, ')')
        else:
            raise SyntaxError("Expected factor (identifier, number, true/false, not, or '(')", self.current_token.line, self.current_token.col)

    def _check_declared(self, var_name: str):
        if var_name not in self.declared_vars:
            raise SemanticError(f"Неописанная переменная '{var_name}'", self.current_token.line, self.current_token.col)
def main():
    if len(sys.argv) < 2:
        print("Usage: python lexer.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    lexer = Lexer(input_file)
    lexer.scan()
    lexer.print_tables()
    lexer.print_operation_table()

    # === СИНТАКСИЧЕСКИЙ АНАЛИЗ ===
    # === СИНТАКСИЧЕСКИЙ И СЕМАНТИЧЕСКИЙ АНАЛИЗ ===
    try:
        parser = SyntaxAnalyzer(lexer.tokens, lexer.identifiers)
        parser.parse_program()
        print("\n Синтаксический и семантический анализ успешно завершены!")
    except (SyntaxError, SemanticError) as e:
        print(f"\n ОШИБКА: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()