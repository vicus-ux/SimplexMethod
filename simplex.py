"""
Универсальный двухфазный симплекс-метод с правилом Блэнда для задач max/min.
"""

import sys
from fractions import Fraction
from typing import List, Tuple

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

    A = []
    b = []
    constraints = []

    for raw in lines:
        if '>=' in raw and raw.startswith('x') and '0' in raw:
            continue

        op = '='

        lhs_str, rhs_str = map(str.strip, raw.split(op, 1))
        lhs = [FractionWithPrint(x) for x in lhs_str.split()]
        rhs = FractionWithPrint(rhs_str)

    
        A.append(lhs)
        b.append(rhs)
        constraints.append('=')

    return A, b, c, goal, constraints


class SimplexSolver:
    """Класс для решения задач линейного программирования двухфазным симплекс-методом."""
    
    def __init__(self, A: Matrix, b: List[Fraction], c: List[Fraction], goal: str, constraints: List[str]):
        self.A = A
        self.b = b
        self.original_c = c
        self.goal = goal
        self.constraints = constraints
        self.n = max(len(row) for row in A) if A else len(c)
        self.m = len(A)
       
        self.c = c.copy() 
        self.tab = []
        self.basis = []
        self.iter_count = 0
        
    def print_table(self, phase: int = 1):
        """Печать симплекс-таблицы."""
        print(f'\n----- Фаза {phase}, Итерация {self.iter_count} -----')
        
        headers = []

        if phase == 1: 
            for i in range(self.n):
                headers.append(f'x{i+1}')
            for i in range(self.m):
                headers.append(f'x_a{self.n+i+1}')
            
            headers.append('b')
            
            print('Basis | ' + ' | '.join(f'{h:>8}' for h in headers))
            print('-' * (14 + 10 * len(headers)))
            
            for i in range(self.m):
                basis_var = self.get_var_name(self.basis[i])
                row_str = f'{basis_var:>5} | '
                row_str += ' | '.join(f'{str(self.tab[i][j]):>8}' for j in range(len(self.tab[i])))
                print(row_str)
            
            obj_name = 'ЦФ'
            print(f'{obj_name:>5} | ' + ' | '.join(f'{str(self.tab[self.m][j]):>8}' for j in range(len(self.tab[self.m]))))

        else:
            for i in range(self.n):
                headers.append(f'x{i+1}')
            headers.append('b')
            
            print('Basis | ' + ' | '.join(f'{h:>8}' for h in headers))
            print('-' * (12 + 10 * len(headers)))
            
            for i in range(self.m):
                basis_var = self.get_var_name(self.basis[i])
                row_str = f'{basis_var:>5} | '
                row_str += ' | '.join(f'{str(self.tab[i][j]):>8}' for j in range(len(self.tab[i])))
                print(row_str)
        
            obj_name = 'ЦФ'
            last_row_index = len(self.tab) - 1
            print(f'{obj_name:>5} | ' + ' | '.join(f'{str(self.tab[last_row_index][j]):>8}' for j in range(len(self.tab[last_row_index]))))

    
    def get_var_name(self, index: int) -> str:
        """Получить имя переменной по индексу."""
        if index < self.n:
            return f'x{index+1}'
        elif index < self.n + self.m:
            return f'x_a{index+1}'
    
    def find_pivot(self) -> Tuple[int, int]:
        """Найти опорный элемент по правилу Блэнда (выбираем наименьший индекс входящей переменной)."""
        pivot_col = -1
        #находим первый отрицательный элемент (выбираем столбец - эта переменная, которая войдет в базис)
        for j in range(len(self.tab[0]) - 1): 
            if self.tab[self.m][j] < 0:
                pivot_col = j
                break
        
        if pivot_col == -1:
            return -1, -1  
            
        valid_rows = []
        for i in range(self.m):
            if self.tab[i][pivot_col] > 0:
                valid_rows.append(i)
        
        if not valid_rows:
            return -1, -1  
            
        min_ratio = None
        min_ratio_rows = []
        
        #вычисляем минимальный коэффициент, между выбранным столбцом и слотвцом b
        for i in valid_rows:
            if self.tab[i][pivot_col] > 0:
                ratio = self.tab[i][-1] / self.tab[i][pivot_col]
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    min_ratio_rows = [i]
                elif ratio == min_ratio:
                    min_ratio_rows.append(i)
        
        #среди строк с минимальным отношением выбираем с минимальным индексом базисной переменной (правило Блэнда)
        pivot_row = min(min_ratio_rows, key=lambda i: self.basis[i])
        
        return pivot_row, pivot_col
    
    def has_negative_in_obj(self) -> bool:
        """True, если можно улучшить ЦФ."""
        return any(self.tab[self.m][j] < 0 for j in range(len(self.tab[0]) - 1))
    
    def iterate(self, pivot_row: int, pivot_col: int):
        """Выполнить одну итерацию симплекс-метода."""
        pivot_val = self.tab[pivot_row][pivot_col]
        
        print(f"Ведущий элемент: строка {pivot_row+1} ({self.get_var_name(self.basis[pivot_row])}), "
              f"столбец {pivot_col} ({self.get_var_name(pivot_col)}), значение: {pivot_val}")
        
        #нормализуем ведущую строку
        for j in range(len(self.tab[pivot_row])):
            self.tab[pivot_row][j] /= pivot_val
        
        #обновляем остальные строки
        for i in range(len(self.tab)):
            if i != pivot_row:
                factor = self.tab[i][pivot_col]
                for j in range(len(self.tab[i])):
                    self.tab[i][j] -= factor * self.tab[pivot_row][j]
        
        self.basis[pivot_row] = pivot_col
        self.iter_count += 1
    
    def previous_phase(self):
        """Подготовительная фаза, перед первой итерацией фазы 1"""
        print("\n--- Вычитание единиц из целевой функции ---")
        

        for i in range(self.m):
            basis_col = self.basis[i]  
            if self.tab[self.m][basis_col] == 1:
                for j in range(len(self.tab[self.m])):
                    self.tab[self.m][j] -= self.tab[i][j]
                
                print(f"Вычтена строка {i+1} ({self.get_var_name(self.basis[i])}) из целевой функции")
        
        print("Целевая функция после вычитания:")
        obj_name = 'ЦФ'
        row_str = f'{obj_name:>5} | '
        row_str += ' | '.join(f'{str(self.tab[self.m][j]):>8}' for j in range(len(self.tab[self.m])))
        print(row_str)

    def phase1(self) -> bool:
        print("\n" + "="*60)
        print("ФАЗА 1: Поиск допустимого базиса")
        print("="*60)

        self.tab = []
        self.basis = []

        #согздание начальной симплекс-таблицы
        for i in range(self.m):
            row = self.A[i].copy()
                 
            row.extend([FractionWithPrint(1) if j == i else FractionWithPrint(0) for j in range(self.m)])
            self.basis.append(self.n + i)
               
            if self.b[i] < 0:
                    row = [-x for x in row]
                    self.b[i] = -self.b[i]
 
            row.append(self.b[i])
            self.tab.append([FractionWithPrint(x) for x in row])

        #целевая функция
        z_row = [FractionWithPrint(0) for _ in range(self.n + self.m + 1)] 

        for i in range(self.m):
            basis_col = self.basis[i] 
            z_row[basis_col] = FractionWithPrint(1) 

        self.tab.append(z_row)
        self.iter_count = 0
        self.print_table(phase=1)
       
        self.previous_phase()

        while self.has_negative_in_obj():
            r, c = self.find_pivot()
            if r == -1:
                print("Неограничено в фазе 1")
                return False
            self.iterate(r, c)
            self.print_table(phase=1)

        print("Фаза 1 завершена\n")
        return True
    
    def iterate_phase2(self):
        """Обнуляет коэффициенты в целевой функции над базисными переменными."""
        print("\n--- Обнуление коэффициентов над базисными переменными ---")
        
        for i in range(self.m):
            basis_col = self.basis[i] 
            basis_coeff = self.tab[self.m][basis_col]  
            
            if basis_coeff != 0:
                factor = basis_coeff  
                
                for j in range(len(self.tab[self.m])):
                    self.tab[self.m][j] -= factor * self.tab[i][j]
                
                print(f"Вычтена строка {i+1} ({self.get_var_name(self.basis[i])}) из целевой функции с множителем {factor}")
        
        print("Целевая функция после обнуления коэффициентов над базисными переменными:")
        obj_name = 'ЦФ'
        row_str = f'{obj_name:>5} | '
        row_str += ' | '.join(f'{str(self.tab[self.m][j]):>8}' for j in range(len(self.tab[self.m])))
        print(row_str)

    def phase2(self) -> Tuple[Fraction, List[Fraction]]:
        """Фаза 2: решение исходной задачи."""
        print("\n" + "="*60)
        print("ФАЗА 2: Решение исходной задачи")
        print("="*60)
        
        #урезаем таблицу (благодарим прошлую матрицу за участие)
        for i in range(self.m):
            self.tab[i] = self.tab[i][:self.n] + [self.tab[i][-1]]
    
        self.tab = self.tab[:self.m] 

        z_row = [FractionWithPrint(0) for _ in range(self.n + 1)]  
        
        for j in range(min(self.n, len(self.c))):  
            z_row[j] = self.c[j]
    
        self.tab.append(z_row) 
        self.tab[self.m] = z_row
        self.iter_count = 0

        print("Начальная таблица фазы 2:")
        self.print_table(phase=2)

        self.iterate_phase2()

        x = [FractionWithPrint(0)] * self.n

        for i, basis_var in enumerate(self.basis):
            if basis_var < self.n:
                x[basis_var] = self.tab[i][-1]
            
        val = FractionWithPrint(0)
        for ci, xi in zip(self.original_c, x):
            val += ci * xi

        obj_value = val
        return obj_value, x

    
    def solve(self) -> Tuple[Fraction, List[Fraction], int]:
        """Решить задачу двухфазным симплекс-методом."""
        total_iters = 0
        
        # Фаза 1
        if not self.phase1():
            raise ValueError("Задача не имеет допустимых решений")
        total_iters += self.iter_count
        
        # Фаза 2
        obj_value, x = self.phase2()
        total_iters += self.iter_count
        
        return obj_value, x, total_iters


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
        opt, x, iters = solver.solve()
        
        print('\n' + "=" * 60)
        print("РЕЗУЛЬТАТ")
        print("=" * 60)
        print(f'Оптимальное значение ЦФ: {opt}')
        print('Оптимальный план:')
        for i, xi in enumerate(x):
            print(f'  x{i+1} = {xi}')
        print(f'Всего итераций: {iters}')

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()