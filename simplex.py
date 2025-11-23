#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симплекс-метод с правилом Бленда (minimal-index) для рациональных дробей.
"""

import sys
from fractions import Fraction
from typing import List

Matrix = List[List[Fraction]]


# ------------------------------------------------------------------
# чтение файла
# ------------------------------------------------------------------
def read_task(filename: str):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError('Файл пуст')
    goal = lines.pop(0).lower()
    if goal not in {'min', 'max'}:
        raise ValueError('Первая строка должна быть min или max')

    c = [Fraction(x) for x in lines.pop(0).split()]
    n = len(c)

    # убираем строки x... >= 0
    lines = [ln for ln in lines if not (ln.startswith('x') and ln.endswith('>= 0'))]

    A: List[List[Fraction]] = []
    b: List[Fraction] = []

    for row in lines:
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
            raise ValueError('Число коэффициентов не совпадает с n')
        if op == '<=':
            A.append(lhs)
            b.append(rhs)
        elif op == '>=':
            A.append([-x for x in lhs])
            b.append(-rhs)
        elif op == '=':
            A.append(lhs)
            b.append(rhs)
            A.append([-x for x in lhs])
            b.append(-rhs)

    if not A:
        raise ValueError('Нет ограничений')
    
    return A, b, c, goal, n


# ------------------------------------------------------------------
# печать
# ------------------------------------------------------------------
def print_matrix(mat: Matrix, width=8):
    for row in mat:
        print(''.join(f'{str(x):>{width}}' for x in row))


# ------------------------------------------------------------------
# симплекс
# ------------------------------------------------------------------
def simplex_blend(A: Matrix, b: List[Fraction], c: List[Fraction], goal: str, n: int):
    m = len(A)
    
    # Создаем таблицу: [A | I | b]
    tab = []
    for i in range(m):
        row = A[i].copy()  # Коэффициенты при x
        # Добавляем единичную матрицу для slack переменных
        for j in range(m):
            row.append(Fraction(1) if i == j else Fraction(0))
        row.append(b[i])  # RHS
        tab.append(row)
    
    # Последняя строка: коэффициенты целевой функции
    # Для симплекс-метода в канонической форме: Z - c^T x = 0
    last_row = []
    for coeff in c:
        last_row.append(-coeff)  # -c для исходных переменных
    
    # Коэффициенты для slack переменных
    for j in range(m):
        last_row.append(Fraction(0))
    
    last_row.append(Fraction(0))  # RHS для Z-строки
    tab.append(last_row)
    
    basis = list(range(n, n + m))  # начальный базис: slack переменные
    it = 0

    def print_iter():
        nonlocal it
        it += 1
        print(f'\n===== Итерация {it} =====')
        print('Базис:', [f'x{i+1}' if i < n else f's{i-n+1}' for i in basis])
        
        # Заголовки для отладки
        headers = [f'x{i+1}' for i in range(n)] + [f's{i+1}' for i in range(m)] + ['RHS']
        print(' '.join(f'{h:>8}' for h in headers))
        
        for row in tab:
            print(' '.join(f'{str(x):>8}' for x in row))
        
        # Вычисляем текущие значения переменных
        x = [Fraction(0)] * (n + m)
        for i, bv in enumerate(basis):
            if bv < n + m:
                x[bv] = tab[i][-1]
        
        print('Текущие x =', [str(xi) for xi in x[:n]])
        
        # Вычисляем значение целевой функции
        if goal == 'max':
            obj = -tab[-1][-1]
        else:
            obj = tab[-1][-1]
        print('ЦФ =', obj)

    print_iter()

    while True:
        # Проверяем оптимальность - ИСПРАВЛЕННАЯ ЛОГИКА
        optimal = True
        for j in range(n + m):
            if tab[-1][j] < 0:  # Упрощенная проверка: если есть отрицательные - не оптимально
                optimal = False
                break
        
        if optimal:
            break

        # Выбор ведущего столбца - ИСПРАВЛЕННАЯ ЛОГИКА
        col = None
        # Для обеих задач выбираем первый столбец с отрицательным коэффициентом
        for j in range(n + m):
            if tab[-1][j] < 0:
                col = j
                break
        
        if col is None:
            break  # Достигнут оптимум

        # Выбор ведущей строки (минимальное положительное отношение)
        row, best_theta = None, None
        for i in range(m):
            if tab[i][col] > 0:
                theta = tab[i][-1] / tab[i][col]
                if best_theta is None or theta < best_theta or (theta == best_theta and i < row):
                    best_theta, row = theta, i
        
        if row is None:
            raise ValueError('ЦФ не ограничена')

        # Пересчет таблицы
        pivot = tab[row][col]
        
        # Нормализуем ведущую строку
        for j in range(len(tab[row])):
            tab[row][j] /= pivot
        
        # Обновляем остальные строки
        for i in range(len(tab)):
            if i != row:
                factor = tab[i][col]
                for j in range(len(tab[i])):
                    tab[i][j] -= factor * tab[row][j]
        
        # Обновляем базис
        basis[row] = col
        print_iter()

    # Вычисляем результат
    x = [Fraction(0)] * (n + m)
    for i, bv in enumerate(basis):
        if bv < len(x):
            x[bv] = tab[i][-1]
    
    if goal == 'max':
        opt = -tab[-1][-1]
    else:
        opt = tab[-1][-1]
    
    return opt, x[:n], it


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print('Usage: python simplex.py task.txt')
        sys.exit(1)
    A, b, c, goal, n = read_task(sys.argv[1])
    opt, x, iters = simplex_blend(A, b, c, goal, n)
    print('\n===== РЕЗУЛЬТАТ =====')
    print('Оптимальное значение ЦФ:', opt)
    print('Оптимальный план x =', [str(xi) for xi in x])
    print('Всего итераций:', iters)


if __name__ == '__main__':
    main()