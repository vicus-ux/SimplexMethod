"""
Универсальный двухфазный симплекс-метод с правилом Блэнда для задач max/min.
"""

import sys
from fractions import Fraction
from typing import List, Tuple, Optional

Matrix = List[List[Fraction]]


class FractionWithPrint(Fraction):
    """Класс дроби с улучшенным выводом."""
    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


def read_task(filename: str) -> Tuple[Matrix, List[Fraction], List[Fraction], str, List[str]]:
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError('Файл пуст')

    goal = lines.pop(0).lower()
    if goal not in {'min', 'max'}:
        raise ValueError('Первая строка должна быть min или max')

    c = [FractionWithPrint(x) for x in lines.pop(0).split()]
    n = len(c)

    A = []
    b = []
    constraints = []

    for raw in lines:
        # Убираем x >= 0
        if '>=' in raw and raw.startswith('x') and '0' in raw:
            continue

        op = None
        for cand in ('<=', '>=', '='):
            if cand in raw:
                op = cand
                break
        if not op:
            continue

        lhs_str, rhs_str = map(str.strip, raw.split(op, 1))
        lhs = [FractionWithPrint(x) for x in lhs_str.split()]
        rhs = FractionWithPrint(rhs_str)

        if len(lhs) != n:
            raise ValueError(f'Число коэффициентов {len(lhs)} не совпадает с n={n}')

        if op == '<=':
            A.append(lhs)
            b.append(rhs)
            constraints.append('<=')
        elif op == '>=':
            A.append([-x for x in lhs])
            b.append(-rhs)
            constraints.append('>=')
        elif op == '=':
            A.append(lhs)
            b.append(rhs)
            constraints.append('=')

    if not A:
        raise ValueError('Нет ограничений')

    return A, b, c, goal, constraints


class SimplexSolver:
    """Класс для решения задач линейного программирования двухфазным симплекс-методом."""
    
    def __init__(self, A: Matrix, b: List[Fraction], c: List[Fraction], goal: str, constraints: List[str]):
        self.A = A
        self.b = b
        self.original_c = c
        self.goal = goal
        self.constraints = constraints
        self.n = len(c)  # количество переменных
        self.m = len(A)  # количество ограничений
        
        # Для максимизации меняем знак целевой функции (работаем затем в форме минимизации)
        if goal == 'max':
            self.c = [-x for x in c]
        else:
            self.c = c.copy()
            
        self.tab = []
        self.basis = []
        self.iter_count = 0
        
    def print_table(self, phase: int = 1):
        """Печать симплекс-таблицы."""
        print(f'\n----- Фаза {phase}, Итерация {self.iter_count} -----')
        
        # Заголовки
        headers = []
        for i in range(self.n):
            headers.append(f'x{i+1}')
        for i in range(self.m):
            headers.append(f's{i+1}')
        if phase == 1:
            for i in range(self.m):
                headers.append(f'a{i+1}')
        headers.append('RHS')
        
        print('Basis | ' + ' | '.join(f'{h:>8}' for h in headers))
        print('-' * (8 + 10 * len(headers)))
        
        # Строки ограничений
        for i in range(self.m):
            basis_var = self.get_var_name(self.basis[i])
            row_str = f'{basis_var:>5} | '
            row_str += ' | '.join(f'{str(self.tab[i][j]):>8}' for j in range(len(self.tab[i])))
            print(row_str)
        
        # Целевая строка
        obj_name = 'w' if phase == 1 else 'z'
        print(f'{obj_name:>5} | ' + ' | '.join(f'{str(self.tab[self.m][j]):>8}' for j in range(len(self.tab[self.m]))))
    
    def get_var_name(self, index: int) -> str:
        """Получить имя переменной по индексу."""
        if index < self.n:
            return f'x{index+1}'
        elif index < self.n + self.m:
            return f's{index - self.n + 1}'
        else:
            return f'a{index - self.n - self.m + 1}'
    
    def find_pivot(self) -> Tuple[int, int]:
        """Найти опорный элемент по правилу Блэнда (выбираем наименьший индекс входящей переменной)."""
        # Поиск ведущего столбца: по правилу Блэнда возьмём наименьший j с отрицательной reduced cost
        pivot_col = -1
        for j in range(len(self.tab[0]) - 1):  # исключаем RHS
            if self.tab[self.m][j] < 0:
                pivot_col = j
                break
        
        if pivot_col == -1:
            return -1, -1  # оптимально
            
        # Поиск допустимых строк для pivot_col
        valid_rows = []
        for i in range(self.m):
            if self.tab[i][pivot_col] > 0:
                valid_rows.append(i)
        
        if not valid_rows:
            return -1, -1  # неограничено
            
        # Выбор ведущей строки по правилу минимального отношения (RHS / a_ij)
        min_ratio = None
        min_ratio_rows = []
        
        for i in valid_rows:
            if self.tab[i][pivot_col] > 0:
                ratio = self.tab[i][-1] / self.tab[i][pivot_col]
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    min_ratio_rows = [i]
                elif ratio == min_ratio:
                    min_ratio_rows.append(i)
        
        # Среди строк с минимальным отношением выбираем с минимальным индексом базисной переменной (правило Блэнда)
        pivot_row = min(min_ratio_rows, key=lambda i: self.basis[i])
        
        return pivot_row, pivot_col
    
    def has_negative_in_obj(self) -> bool:
        """True, если можно улучшить ЦФ (поиск отрицательных reduced costs)."""
        # После преобразования goal->c (max->-c) мы всегда работаем в форме минимизации,
        # поэтому улучшается цель при наличии отрицательных значений в z/w-строке.
        return any(self.tab[self.m][j] < 0 for j in range(len(self.tab[0]) - 1))
    
    def iterate(self, pivot_row: int, pivot_col: int):
        """Выполнить одну итерацию симплекс-метода."""
        pivot_val = self.tab[pivot_row][pivot_col]
        
        print(f"Ведущий элемент: строка {pivot_row} ({self.get_var_name(self.basis[pivot_row])}), "
              f"столбец {pivot_col} ({self.get_var_name(pivot_col)}), значение: {pivot_val}")
        
        # Нормализуем ведущую строку
        for j in range(len(self.tab[pivot_row])):
            self.tab[pivot_row][j] /= pivot_val
        
        # Обновляем остальные строки
        for i in range(len(self.tab)):
            if i != pivot_row:
                factor = self.tab[i][pivot_col]
                for j in range(len(self.tab[i])):
                    self.tab[i][j] -= factor * self.tab[pivot_row][j]
        
        # Обновляем базис
        self.basis[pivot_row] = pivot_col
        self.iter_count += 1
    
    def phase1(self) -> bool:
        print("\n" + "="*60)
        print("ФАЗА 1: Поиск допустимого базиса")
        print("="*60)

        # Добавляем дополнительные и искусственные переменные
        self.tab = []
        art_cols = 0
        self.basis = []

        for i in range(self.m):
            row = self.A[i].copy()
            # Дополнительные переменные (slack)
            row.extend([FractionWithPrint(1) if j == i else FractionWithPrint(0) for j in range(self.m)])
            # Искусственные только если RHS < 0 или '='
            if self.b[i] < 0 or self.constraints[i] == '=':
                # добавляем столбец(ы) искусственных переменных (максимум m, но используем art_cols счетчик)
                row.extend([FractionWithPrint(1) if j == art_cols else FractionWithPrint(0) for j in range(self.m)])
                self.basis.append(self.n + self.m + art_cols)
                art_cols += 1
                if self.b[i] < 0:
                    row = [-x for x in row]
                    self.b[i] = -self.b[i]
            else:
                # базисная слаковая переменная
                self.basis.append(self.n + i)
            row.append(self.b[i])
            self.tab.append([FractionWithPrint(x) for x in row])

        # Целевая функция: минимизация суммы искусственных
        w_row = [FractionWithPrint(0) for _ in range(self.n + self.m + art_cols + 1)]
        for i in range(self.m):
            if self.basis[i] >= self.n + self.m:
                for j in range(self.n + self.m + art_cols + 1):
                    w_row[j] -= self.tab[i][j]
        self.tab.append(w_row)

        self.iter_count = 0
        self.print_table(phase=1)

        # Итерации фазы 1 (если нужны)
        while self.has_negative_in_obj():
            r, c = self.find_pivot()
            if r == -1:
                print("Неограничено в фазе 1")
                return False
            self.iterate(r, c)
            self.print_table(phase=1)

        # Проверяем значение вспомогательной целевой функции (RHS w)
        if self.tab[self.m][-1] != 0:
            print("✗ Фаза 1: сумма искусственных переменных не равна 0 → допустимого решения нет")
            return False

        # ----- Детализированное объяснение -----
        print("\nПричина завершения фазы 1:")

        # 1) Список искусственных переменных (по базису)
        art_vars = [bv for bv in self.basis if bv >= self.n + self.m]

        if len(art_vars) == 0:
            print("• В задаче нет искусственных переменных → базис допустимый с самого начала.")
        else:
            # Проверка значений искусственных переменных в базисе
            all_zero = True
            for i, bv in enumerate(self.basis):
                if bv >= self.n + self.m:
                    # значение искусственной переменной в строке i
                    val = self.tab[i][-1]
                    print(f"• Искусственная переменная a{bv - (self.n + self.m) + 1} = {val}")
                    if val != 0:
                        all_zero = False
            if all_zero:
                print("• Все искусственные переменные равны 0 → найден допустимый базис.")
            else:
                print("• Некоторые искусственные переменные не нулевые → (это необычная ситуация).")

        print("✓ Фаза 1 завершена\n")
        return True
    
    def phase2(self) -> Tuple[Fraction, List[Fraction], List[Fraction]]:
        """Фаза 2: решение исходной задачи."""
        print("\n" + "="*60)
        print("ФАЗА 2: Решение исходной задачи")
        print("="*60)

        # Удаляем искусственные переменные (если они были добавлены, обрезаем к n+m столбцам + RHS)
        for i in range(self.m):
            self.tab[i] = self.tab[i][:self.n + self.m] + [self.tab[i][-1]]
        # Для z-строки тоже (хотя для w_row она уже корректной длины)
        self.tab[self.m] = self.tab[self.m][:self.n + self.m] + [self.tab[self.m][-1]]


        # Новая целевая строка: z = c^T x (в форме reduced costs r_j = c_j - c_B^T * A_j)
        z_row = [FractionWithPrint(0) for _ in range(self.n + self.m + 1)]

        # Инициализируем z-строку: z_j = c_j (для x) и 0 для slack
        for j in range(self.n + self.m):
            z_row[j] = self.c[j] if j < self.n else FractionWithPrint(0)

        # Пересчитываем z-строку: z_j -= c_B^T * A_j
        for i in range(self.m):
            basis_var = self.basis[i]
            coeff = self.c[basis_var] if basis_var < self.n else FractionWithPrint(0)
            if coeff != 0:
                # корректируем по всей длине (включая RHS позицию)
                for j in range(self.n + self.m + 1):
                    z_row[j] -= coeff * self.tab[i][j]

        self.tab[self.m] = z_row
        self.iter_count = 0

        print("Начальная таблица фазы 2:")
        self.print_table(phase=2)

        # Симплекс-метод (пока есть отрицательные reduced costs)
        while self.has_negative_in_obj():
            pivot_row, pivot_col = self.find_pivot()
            if pivot_row == -1:
                print("Задача неограничена в фазе 2")
                break
            self.iterate(pivot_row, pivot_col)
            self.print_table(phase=2)

        # Извлечение решения
        x = [FractionWithPrint(0)] * self.n
        s = [FractionWithPrint(0)] * self.m

        for i, basis_var in enumerate(self.basis):
            if basis_var < self.n:
                x[basis_var] = self.tab[i][-1]
            elif basis_var < self.n + self.m:
                s[basis_var - self.n] = self.tab[i][-1]

        # Объектное значение: так как мы работаем в форме минимизации (c предварительно изменён),
        # значение в z-строке в RHS — это значение c^T x; возвращаем с учётом исходной цели
        val = sum(c * xi for c, xi in zip(self.original_c, x))
        obj_value = val if self.goal == 'max' else -val
        return obj_value, x, s
    
    def solve(self) -> Tuple[Fraction, List[Fraction], List[Fraction], int]:
        """Решить задачу двухфазным симплекс-методом."""
        total_iters = 0
        
        # Фаза 1
        if not self.phase1():
            raise ValueError("Задача не имеет допустимых решений")
        total_iters += self.iter_count
        
        # Фаза 2
        obj_value, x, s = self.phase2()
        total_iters += self.iter_count
        
        return obj_value, x, s, total_iters


def main():
    if len(sys.argv) != 2:
        print('Использование: python simplex.py task.txt')
        sys.exit(1)
    
    try:
        A, b, c, goal, constraints = read_task(sys.argv[1])
        
        print("=" * 60)
        print("ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД С ПРАВИЛОМ БЛЭНДА")
        print("=" * 60)
        print(f"Тип задачи: {goal}")
        print(f"Коэффициенты ЦФ: {[str(ci) for ci in c]}")
        print(f"Число переменных: {len(c)}")
        print(f"Число ограничений: {len(A)}")
        print("Ограничения:")
        for i, (ai, bi, constr) in enumerate(zip(A, b, constraints)):
            print(f"  {i+1}: {[str(a) for a in ai]} {constr} {bi}")
        
        solver = SimplexSolver(A, b, c, goal, constraints)
        opt, x, s, iters = solver.solve()
        
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
            lhs = sum(A[i][j] * x[j] for j in range(len(c)))
            if constraints[i] == '<=':
                status = lhs <= b[i]
                symbol = '<='
            elif constraints[i] == '>=':
                status = lhs >= b[i]  
                symbol = '>='
            else:  # '='
                status = lhs == b[i]
                symbol = '='
            print(f'  Ограничение {i+1}: {lhs} {symbol} {b[i]} ({status})')
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
