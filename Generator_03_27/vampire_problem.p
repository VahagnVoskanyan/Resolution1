% TPTP problem file generated for Vampire
tff(type_declarations, type,
  P: $i * $i * $i > $o & Q: $i > $o & R: $i > $o & S: $i * $i > $o & T: $i * $i > $o
).

tff(function_declarations, type,
  f: $i > $i & g: $i > $i & h: $i > $i
).

tff(axiom_1, axiom, R(x,w,x) | P(v,w)).
tff(axiom_2, axiom, T(w,w,u) | ~S(x,x) | ~R(y,y)).
tff(axiom_3, axiom, P(z,u,v) | ~T(w,u,z)).
tff(axiom_4, axiom, R(y,w) | ~T(v,u,x)).
tff(axiom_5, axiom, T(z,v,z) | R(v)).
tff(axiom_6, axiom, T(w,w) | P(u,w,w) | T(v,u,v)).
tff(axiom_7, axiom, ~T(z,x,z) | S(u)).
tff(axiom_8, axiom, ~Q(v,y) | ~R(x)).
tff(axiom_9, axiom, S(w,z,x) | ~S(u) | S(w,u,u)).
tff(axiom_10, axiom, ~Q(y,z) | Q(w,u) | S(z,v)).
tff(axiom_11, axiom, Q(x,x) | P(y,u,w)).
tff(axiom_12, axiom, ~S(x,x,x) | ~R(v,w,y)).
tff(axiom_13, axiom, ~T(v) | ~P(z,w) | P(u,x,x)).
tff(axiom_14, axiom, P(u,x) | S(v,v,v)).
tff(axiom_15, axiom, S(v,u,z) | Q(v,y)).
tff(goal, conjecture, P(v,z,u) | P(y,y,x) | P(y,u,u)).