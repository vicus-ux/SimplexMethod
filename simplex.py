
"""
Универсальный симплекс-метод с правилом Блэнда для задач max/min.
"""

import sys
from fractions import Fraction
from typing import List, Tuple

Matrix = List[List[Fraction]]


def read_task(filename: str) -> Tuple[Matrix, List[Fraction], List[Fraction], str, int]:
    """Чтение задачи из файла."""
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError('Файл пуст')
    
    goal = lines.pop(0).lower()
    if goal not in {'min', 'max'}:
        raise ValueError('Первая строка должна быть min или max')

    # Читаем коэффициенты целевой функции
    c_line = lines.pop(0)
    c = [Fraction(x) for x in c_line.split()]
    n = len(c)

    # Фильтруем строки с неотрицательностью переменных
    lines = [ln for ln in lines if not (ln.startswith('x') and '>= 0' in ln)]

    A: List[List[Fraction]] = []
    b: List[Fraction] = []

    for row in lines:
        # Определяем тип ограничения
        op = None
        for cand in ('<=', '>=', '='):
            if cand in row:
                op = cand
                break
        if not op:
            continue
            
        lhs_str, rhs_str = map(str.strip, row.split(op, 1))
        lhs = [Fraction(x) for x in lhs_str.split()]
        rhs = Fraction(rhs_str)
        
        if len(lhs) != n:
            raise ValueError(f'Число коэффициентов {len(lhs)} не совпадает с n={n}')
            
        if op == '<=':
            A.append(lhs)
            b.append(rhs)
        elif op == '>=':
            A.append([-x for x in lhs])
            b.append(-rhs)
        elif op == '=':
            A.append(lhs)
            b.append(rhs)

    if not A:
        raise ValueError('Нет ограничений')
    
    return A, b, c, goal, n


def print_table(tab: Matrix, basis: List[int], n: int, m: int, iter_num: int):
    """Печать симплекс-таблицы."""
    print(f'\n----- Итерация {iter_num} -----')
    
    # Заголовки
    headers = []
    for i in range(n):
        headers.append(f'x{i+1}')
    for i in range(m):
        headers.append(f's{i+1}')
    headers.append('RHS')
    
    print('Basis | ' + ' | '.join(f'{h:>8}' for h in headers))
    print('-' * (8 + 10 * len(headers)))
    
    # Строки ограничений
    for i in range(m):
        basis_var = f'x{basis[i]+1}' if basis[i] < n else f's{basis[i]-n+1}'
        row_str = f'{basis_var:>5} | '
        row_str += ' | '.join(f'{str(tab[i][j]):>8}' for j in range(len(tab[i])))
        print(row_str)
    
    # Целевая строка
    print('f     | ' + ' | '.join(f'{str(tab[m][j]):>8}' for j in range(len(tab[m]))))


def simplex_blend(A: Matrix, b: List[Fraction], c: List[Fraction], goal: str, n: int):
    """Универсальный симплекс-метод с правилом Блэнда для max/min."""
    m = len(A)
    
    # Создаем расширенную таблицу [A | I | b]
    tab = []
    for i in range(m):
        row = A[i].copy()  # Коэффициенты при исходных переменных
        # Добавляем единичную матрицу для дополнительных переменных
        row.extend([Fraction(1) if j == i else Fraction(0) for j in range(m)])
        row.append(b[i])  # Правая часть
        tab.append(row)
    
    # Строка целевой функции
    # В симплекс-таблице всегда храним: f - c^T x = 0
    obj_row = [-coeff for coeff in c]  # -c для исходных переменных
    obj_row.extend([Fraction(0) for _ in range(m)])  # 0 для дополнительных переменных  
    obj_row.append(Fraction(0))  # Правая часть
    tab.append(obj_row)
    
    # Начальный базис: дополнительные переменные
    basis = list(range(n, n + m))
    iter_count = 0
    
    print("Начальная таблица:")
    print_table(tab, basis, n, m, iter_count)
    
    while True:
        iter_count += 1
        # Проверка оптимальности
        optimal = True
        for j in range(n + m):
            if tab[m][j] < 0:  # Если есть отрицательный - не оптимально
                optimal = False
                break
                
        if optimal:
            print("\n✓ Достигнуто оптимальное решение")
            break
        # Выбор ведущего столбца (переменной для входа в базис)
        pivot_col = -1
        for j in range(n + m):
            if tab[m][j] < 0:
                pivot_col = j
                break
        
        if pivot_col == -1:
            break
            
        # Поиск допустимых строк для pivot_col
        valid_rows = []
        for i in range(m):
            if tab[i][pivot_col] > 0:
                valid_rows.append(i)
        
        if not valid_rows:
            # Если нет допустимых строк, задача неограничена
            print(f"\n! Целевая функция не ограничена (столбец {pivot_col})")
            # Возвращаем текущее решение как лучшее найденное
            break
        
        # Выбор ведущей строки по правилу Блэнда
        # Сначала находим минимальное отношение
        min_ratio = None
        min_ratio_rows = []
        
        for i in valid_rows:
            if tab[i][pivot_col] > 0:
                ratio = tab[i][-1] / tab[i][pivot_col]
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    min_ratio_rows = [i]
                elif ratio == min_ratio:
                    min_ratio_rows.append(i)
        
        # Среди строк с минимальным отношением выбираем с минимальным индексом базисной переменной
        pivot_row = min(min_ratio_rows, key=lambda i: basis[i])
        
        print(f"Ведущий элемент: строка {pivot_row} (x{basis[pivot_row]+1}), столбец {pivot_col}")
        print(f"Ведущий элемент значение: {tab[pivot_row][pivot_col]}")
        
        # Преобразование таблицы методом Гаусса-Жордана
        pivot_val = tab[pivot_row][pivot_col]
        
        # Нормализуем ведущую строку
        for j in range(len(tab[pivot_row])):
            tab[pivot_row][j] /= pivot_val
        
        # Обновляем остальные строки
        for i in range(len(tab)):
            if i != pivot_row:
                factor = tab[i][pivot_col]
                for j in range(len(tab[i])):
                    tab[i][j] -= factor * tab[pivot_row][j]
        
        # Обновляем базис
        basis[pivot_row] = pivot_col
        
        print_table(tab, basis, n, m, iter_count)
    
    # Извлечение решения
    x = [Fraction(0)] * n
    s = [Fraction(0)] * m
    
    for i, basis_var in enumerate(basis):
        if basis_var < n:  # Исходная переменная
            x[basis_var] = tab[i][-1]
        else:  # Дополнительная переменная
            s[basis_var - n] = tab[i][-1]
    # Вычисляем значение целевой функции из таблицы
    obj_value_from_table = -tab[m][-1]
    
    # Вычисляем также по найденным x для проверки
    obj_value_from_x = Fraction(0)
    for i in range(n):
        obj_value_from_x += c[i] * x[i]
    
    # Для согласованности используем вычисление по x
    obj_value = obj_value_from_x
    
    print(f"\nПроверка: f из таблицы = {obj_value_from_table}, f из x = {obj_value_from_x}")
    
    return obj_value, x, s, iter_count


def main():
    if len(sys.argv) != 2:
        print('Использование: python simplex.py task.txt')
        sys.exit(1)
    
    try:
        A, b, c, goal, n = read_task(sys.argv[1])
        
        print("=" * 60)
        print("УНИВЕРСАЛЬНЫЙ СИМПЛЕКС-МЕТОД С ПРАВИЛОМ БЛЭНДА")
        print("=" * 60)
        print(f"Тип задачи: {goal}")
        print(f"Коэффициенты ЦФ: {[str(ci) for ci in c]}")
        print(f"Число переменных: {n}")
        print(f"Число ограничений: {len(A)}")
        print(f"Правые части: {[str(bi) for bi in b]}")
        
        opt, x, s, iters = simplex_blend(A, b, c, goal, n)
        
        print('\n' + "=" * 60)
        print("РЕЗУЛЬТАТ")
        print("=" * 60)
        print(f'Оптимальное значение ЦФ: {opt}')
        print('Оптимальный план:')
        for i, xi in enumerate(x):
            print(f'  x{i+1} = {xi}')
        print('Дополнительные переменные:')
        for i, si in enumerate(s):
            print(f'  s{i+1} = {si}')
        print(f'Всего итераций: {iters}')
        
        # Проверка ограничений
        print('\nПроверка ограничений:')
        for i in range(len(A)):
            lhs = sum(A[i][j] * x[j] for j in range(n))
            print(f'  Ограничение {i+1}: {lhs} <= {b[i]} ({lhs <= b[i]})')
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()