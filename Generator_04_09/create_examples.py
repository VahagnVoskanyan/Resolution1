import random, math
import numpy as np
import json

import run_comnnads_solve
from unification_resolution import UnificationResolution
from create_examples_helpers import negate_literal, parse_tptp_clauses, write_to_tptp


def all_resolvents(C1, C2):
    """
    Возвращает список [ new_clause, ... ] — результат резолюции между C1 и C2.
    Предполагаем, что C1, C2 — множества литералов вида { L1, L2, ... }.
    Игнорируем superposition: '=' трактуем как обычный предикат eq(t1,t2).
    """
    resolvents = []

    resolver = UnificationResolution()
    for litA in C1:
        for litB in C2:
            unifier = resolver.can_resolve(litA, litB)
            if unifier is None:
                continue
                
            # Build the resolvent clause by applying substitutions
            resolvent = set()
            
            # Add literals from clauseA except litA
            for lit in C1:
                if lit != litA:
                    resolvent.add(resolver.apply_subst_to_literal(lit, unifier))
            
            # Add literals from clauseB except litB
            for lit in C2:
                if lit != litB:
                    resolvent.add(resolver.apply_subst_to_literal(lit, unifier))
            
            resolvents.append(resolvent)

    return resolvents

def forward_propose(axioms, N=10, temperature=5.0):
    """
    Генерирует линейную цепочку C0..C_N и возвращает последнюю клаузу C_N.
    
    :param axioms: список троек (name, role, set_of_literals), полученных из parse_tptp_clauses
    :param N: количество шагов (глубина цепочки)
    :param temperature: параметр для softmax-взвешивания по размеру клаузы
    :return: конечная клауза (набор литералов), которая будет выступать в качестве утверждения (conjecture)
    """
    # Берём все аксиомные клаузы (роль 'axiom')
    known_clauses = [cl for (_, role, cl) in axioms if role == 'axiom']
    
    # Выбираем случайно стартовую клаузу C0
    C0 = random.choice(known_clauses)
    chain = [C0]

    for t in range(N):
        candidates = []
        # Для каждой известной клаузы пытаемся резолвировать chain[-1] с ней
        for c2 in known_clauses:
            cand = all_resolvents(chain[-1], c2)
            candidates.extend(cand)
        
        # Если ни одной резолвенты не найдено, цепочка остановлена
        if not candidates:
            break
        
        # Softmax по «размеру» (числу литералов) – чем меньше клауза, тем выше вероятность выбора
        sizes = np.array([len(c) for c in candidates])
        weights = np.exp(-sizes / temperature)
        probs = weights / np.sum(weights)
        idx = np.random.choice(len(candidates), p=probs)
        Cnew = candidates[idx]
        
        chain.append(Cnew)
        known_clauses.append(Cnew)
    
    # Возвращаем последнюю клаузу цепочки, которая будет нашим «conjecture»
    return chain[-1]

def negate_clause(clause: set) -> list:
    """
    Inverts a disjunction:
      C = (L1 ∨ L2 ∨ ... )
      ¬C = (¬L1) ∧ (¬L2) ∧ ...
    Returns a list of new TPTP-style clauses, where each is a singleton set containing the negated literal.
    """
    negated_clauses = []
    for L in clause:
        neg_literal = negate_literal(L)
        negated_clauses.append({neg_literal})
    return negated_clauses

def generate_problem(axioms, N=10, T=5.0):
    """
    Генерируем одну задачу: axioms + neg(C_N).
    Возвращает список клауз (axiom + negated_conjecture).
    """
    C_final = forward_propose(axioms, N, T)
    negated = negate_clause(C_final)
    
    # Собираем всё вместе
    problem = []
    for (nm,role,cl) in axioms:
        if role=='axiom':
            problem.append((nm, 'axiom', cl))
    i = 0
    for ncl in negated:
        i += 1
        problem.append((f'goal_{i}', 'negated_conjecture', ncl))
    
    return problem

# #Пример использования:
# axioms = parse_tptp_clauses('Axioms_clausified/NUM005+1.ax_claused.txt')
# for k in range(1000):
#     prob = generate_problem(axioms, N=10, T=5.0)
#     write_to_tptp(prob, f'gen_problem_{k}.p')

# #Пример использования: parse_tptp_clauses(filename)
# if __name__ == '__main__':
#     filePath = 'Axioms_clausified/CAT002-0.ax_claused.txt'
#     cl_list = parse_tptp_clauses(filePath)
#     for cl in cl_list:
#         print(cl)

if __name__ == '__main__':
    #fileName = "CAT001-0"
    #fileName = "CAT002-0"
    #fileName = "CAT003-0"
    fileName = "CAT004-0"
    axioms = parse_tptp_clauses(f'Axioms_clausified/{fileName}.ax_claused.txt')

    for k in range(10000):
        N=10
        T=8.0
        problem = generate_problem(axioms, N, T)

        from generate_dataset import find_candidate_resolvable_pairs
        resolvable_pairs = find_candidate_resolvable_pairs(problem)

        json_filename = f'Res_Pairs/gen_problem_{fileName}_N={N}_T={T}_{k}_rs.jsonl'

        with open(json_filename, "w") as fp:
            fp.write(json.dumps(resolvable_pairs) + "\n")
        print(f"wrote {json_filename}")

        # with open(json_filename, 'w') as fp:
        #     fp.write(json.dumps({"clauses": resolvable_pairs["clauses"]}) + "\n")
        #     for candidate in resolvable_pairs["resolvable_pairs"]:
        #         fp.write(json.dumps(candidate) + "\n")
        #     fp.write(json.dumps({"best_pair": resolvable_pairs["best_pair"]}))
            #json.dump(resolvable_pairs, fp)

        ###
        # from generate_dataset import serializa_clauses
        # ser_clauses = serializa_clauses(problem)

        # json_filename = f'dataset/gen_problem_clauses_{fileName}_N={N}_T={T}_{k}.jsonl'

        # with open(json_filename, 'w') as fp:
        #      json.dump(ser_clauses, fp)
        ###

        write_to_tptp(problem, f'Gen_Problems/gen_problem_{fileName}_N={N}_T={T}_{k}.p')

        # Slows down everything
        #run_comnnads_solve.run_docker_solve_command()

