cnf(u1, axiom, (there_exists(X0) | ~equivalent(X0,X1))).
cnf(u2, axiom, (X0 = X1 | ~equivalent(X0,X1))).
cnf(u18, axiom, (equivalent(X1,X1) | ~there_exists(X1))).
cnf(u4, axiom, (there_exists(X0) | ~there_exists(domain(X0)))).
cnf(u5, axiom, (there_exists(X0) | ~there_exists(codomain(X0)))).
cnf(u6, axiom, (there_exists(domain(X0)) | ~there_exists(compose(X0,X1)))).
cnf(u7, axiom, (domain(X0) = codomain(X1) | ~there_exists(compose(X0,X1)))).
cnf(u8, axiom, (domain(X0) != codomain(X1) | there_exists(compose(X0,X1)) | ~there_exists(domain(X0)))).
cnf(u9, axiom, (compose(X0,compose(X1,X2)) = compose(compose(X0,X1),X2))).
cnf(u10, axiom, (compose(X0,domain(X0)) = X0)).
cnf(u11, axiom, (compose(codomain(X0),X0) = X0)).
cnf(u12, axiom, (there_exists(X1) | ~equivalent(X0,X1))).
cnf(u20, axiom, (equivalent(X1,X1) | ~there_exists(X1))).
cnf(u14, axiom, (there_exists(codomain(X0)) | ~there_exists(compose(X0,X1)))).
cnf(u15, axiom, (X0 = X1 | there_exists(f1(X0,X1)))).
cnf(u16, axiom, (X0 = X1 | f1(X0,X1) = X0 | f1(X0,X1) = X1)).
cnf(u17, axiom, (X0 = X1 | f1(X0,X1) != X0 | f1(X0,X1) != X1)).
cnf(goal_1, negated_conjecture, (equivalent(X0,domain(X0)))).
cnf(goal_2, negated_conjecture, (domain(X0) = codomain(domain(X0)))).
cnf(goal_3, negated_conjecture, (~equivalent(codomain(X0),codomain(X0)))).
