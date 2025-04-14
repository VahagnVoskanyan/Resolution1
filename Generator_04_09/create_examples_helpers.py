import re

from unification_resolution import UnificationResolution

def parse_tptp_clauses(filePath):
    """
    Простейший парсер TPTP для axiom-клауз.
    Возвращает список троек (name, role, set_of_literals).
    Каждый литерал представлен в виде строки.
    """
    clauses = []
    with open(filePath, 'r') as f:
        # Читаем файл целиком и удаляем комментарии (начинающиеся с %)
        content = f.read()
    # Удаляем строки-комментарии
    content = "\n".join(line for line in content.splitlines() if not line.strip().startswith('%'))
    
    # Поскольку каждая клауза оканчивается на ").", используем регекс чтобы найти все конструкции
    # Примерная форма: cnf(u65,axiom, ... ).  (с возможными переносами строк)
    pattern = r'cnf\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,(.*?)\)\s*\.'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for m in matches:
        name = m.group(1).strip()
        role = m.group(2).strip()
        formula = m.group(3).strip()
        
        # Если формула окружена внешними скобками, удалим их
        if formula.startswith('(') and formula.endswith(')'):
            formula = formula[1:-1].strip()
        
        # Разбиваем формулу по символу '|' для получения литералов.
        # Здесь предполагаем, что оператор дисъюнкции не встречается внутри литералов.
        literals = [lit.strip() for lit in formula.split('|') if lit.strip()]
        # Приводим к множеству (или можно оставить списком, если важна кратность)
        literal_set = set(literals)
        
        clauses.append((name, role, literal_set))
    
    return clauses

def negate_literal(literal: str) -> str:
    """
    Negate a literal:
    - For non-equality predicates: if the literal is positive, add '~'; if it's negative (starts with '~'),
      remove the '~'.
    - For equality, if literal is positive (format "X = Y"), produce negative form ("X != Y"), and vice versa.
    """
    resolver = UnificationResolution()
    sign, pred, args = resolver.parse_literal(literal)
    if pred == "eq":
        if sign > 0:
            # Positive equality becomes inequality.
            return f"{args[0]} != {args[1]}"
        else:
            # Negative equality becomes positive equality.
            return f"{args[0]} = {args[1]}"
    else:
        if sign > 0:
            # Positive literal: add a negation.
            if args:
                return f"~{pred}({', '.join(args)})"
            else:
                return f"~{pred}"
        else:
            # Negative literal: remove the negation.
            if args:
                return f"{pred}({', '.join(args)})"
            else:
                return pred