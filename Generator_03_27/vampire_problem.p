% Problem: resolution_problem_6070
tff(predicates, type, Pred1: $i * $i > $o & Pred2: $i > $o & Pred3: $i > $o).
tff(functions, type, func_f: $i > $i & func_g: $i > $i & func_h: $i > $i).
tff(axiom_1, axiom, (Pred2(Z,Z) | ~Pred1(X,Y))).
tff(axiom_2, axiom, (~Pred1(Y) | ~Pred3(Y,Y))).
tff(axiom_3, axiom, (~Pred3(Y,X) | ~Pred3(Z,Y) | ~Pred1(Y,Z))).
tff(axiom_4, axiom, (Pred2(Y,Y) | Pred3(X,Z))).
tff(axiom_5, axiom, (~Pred2(Z,Y) | Pred3(X) | Pred2(X,X))).
tff(axiom_6, axiom, (Pred3(Z,Z) | Pred2(Z) | Pred2(X,Z))).
tff(axiom_7, axiom, (~Pred3(Z) | Pred2(X,X))).
tff(axiom_8, axiom, (~Pred1(X,Y) | Pred2(X,X) | Pred2(X,Z))).
tff(axiom_9, axiom, (Pred1(Z,Y) | ~Pred1(Z,X) | ~Pred2(X,Z))).
tff(axiom_10, axiom, (Pred3(Y) | ~Pred3(Y,X))).
tff(axiom_11, axiom, (~Pred3(Y,Z) | Pred1(X,Y))).
tff(axiom_12, axiom, (~Pred3(X,Y) | ~Pred2(Z,Z) | Pred3(X,X))).
tff(axiom_13, axiom, (~Pred2(X,Z) | Pred3(X) | Pred1(Z,Y))).
tff(axiom_14, axiom, (~Pred1(X) | Pred1(Y,Z))).
tff(axiom_15, axiom, (Pred1(Z,Z) | ~Pred2(X,Y) | ~Pred1(Y))).
tff(goal, conjecture, (Pred1(Y) | ~Pred1(Z,Y))).
