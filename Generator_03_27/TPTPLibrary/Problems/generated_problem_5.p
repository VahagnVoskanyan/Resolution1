%------------------------------------------------------------------------------
% File     : AGT001+0 : TPTP v9.0.0. Released v2.7.0.
% Domain   : Agents
% Axioms   : CPlanT system
% Version  : [Bar03] axioms : Especial.
% English  :

% Refs     : [Bar03] Barta, J. (2003), Email to G. Sutcliffe
% Source   : [Bar03]
% Names    :

% Status   : Satisfiable
% Syntax   : Number of formulae    :   20 (   0 unt;   0 def)
%            Number of atoms       :   98 (   0 equ)
%            Maximal formula atoms :    6 (   4 avg)
%            Number of connectives :   79 (   1   ~;   0   |;  58   &)
%                                         (  14 <=>;   6  =>;   0  <=;   0 <~>)
%            Maximal formula depth :    8 (   7 avg)
%            Maximal term depth    :    1 (   1 avg)
%            Number of predicates  :   10 (  10 usr;   0 prp; 2-4 aty)
%            Number of functors    :   47 (  47 usr;  47 con; 0-0 aty)
%            Number of variables   :   35 (  35   !;   0   ?)
% SPC      : 

% Comments : Requires NUM005+0.ax NUM005+1.ax
%------------------------------------------------------------------------------
fof(a1_1,axiom,
    ! [A,C,N,L] :
      ( accept_team(A,L,C,N)
    <=> ( accept_city(A,C)
        & accept_leader(A,L)
        & accept_number(A,N) ) ) ).

fof(a1_2,axiom,
    ! [A,N,M] :
      ( ( accept_number(A,N)
        & less(M,N) )
     => accept_number(A,M) ) ).

fof(a1_3,axiom,
    ! [A,N,M,P] :
      ( ( accept_population(A,P,N)
        & less(M,N) )
     => accept_population(A,P,M) ) ).

fof(a1_4,axiom,
    ! [A,L,C] :
      ( the_agent_in_all_proposed_teams(A,L,C)
     => ( accept_leader(A,L)
        & accept_city(A,C) ) ) ).

fof(a1_5,axiom,
    ! [A,L,C] :
      ( any_agent_in_all_proposed_teams(A,L,C)
     => accept_leader(A,L) ) ).

fof(a1_6,axiom,
    ! [A,L,C] :
      ( the_agent_not_in_any_proposed_teams(A,L,C)
     => ~ ( accept_city(A,C)
          & accept_leader(A,L) ) ) ).

fof(a1_7,axiom,
    ! [A,N] :
      ( min_number_of_proposed_agents(A,N)
     => accept_number(A,N) ) ).

fof(a2_1,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n65)
        & accept_population(A,christian,n20)
        & accept_population(A,muslim,n7)
        & accept_population(A,native,n4)
        & accept_population(A,other,n4) )
    <=> accept_city(A,suffertown) ) ).

fof(a2_2,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n54)
        & accept_population(A,christian,n23)
        & accept_population(A,muslim,n3)
        & accept_population(A,native,n1)
        & accept_population(A,other,n9) )
    <=> accept_city(A,centraltown) ) ).

fof(a2_3,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n30)
        & accept_population(A,christian,n8)
        & accept_population(A,muslim,n60)
        & accept_population(A,native,n1)
        & accept_population(A,other,n1) )
    <=> accept_city(A,sunnysideport) ) ).

fof(a2_4,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n70)
        & accept_population(A,christian,n15)
        & accept_population(A,muslim,n1)
        & accept_population(A,native,n10)
        & accept_population(A,other,n4) )
    <=> accept_city(A,centrallakecity) ) ).

fof(a2_5,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n68)
        & accept_population(A,christian,n16)
        & accept_population(A,muslim,n1)
        & accept_population(A,native,n11)
        & accept_population(A,other,n4) )
    <=> accept_city(A,stjosephburgh) ) ).

fof(a2_6,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n70)
        & accept_population(A,christian,n13)
        & accept_population(A,muslim,n0)
        & accept_population(A,native,n15)
        & accept_population(A,other,n2) )
    <=> accept_city(A,northport) ) ).

fof(a2_7,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n12)
        & accept_population(A,christian,n3)
        & accept_population(A,muslim,n0)
        & accept_population(A,native,n85)
        & accept_population(A,other,n0) )
    <=> accept_city(A,coastvillage) ) ).

fof(a2_8,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n0)
        & accept_population(A,christian,n0)
        & accept_population(A,muslim,n0)
        & accept_population(A,native,n100)
        & accept_population(A,other,n0) )
    <=> accept_city(A,sunsetpoint) ) ).

fof(a2_9,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n75)
        & accept_population(A,christian,n24)
        & accept_population(A,muslim,n1)
        & accept_population(A,native,n0)
        & accept_population(A,other,n0) )
    <=> accept_city(A,towna) ) ).

fof(a2_10,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n75)
        & accept_population(A,christian,n25)
        & accept_population(A,muslim,n0)
        & accept_population(A,native,n0)
        & accept_population(A,other,n0) )
    <=> accept_city(A,citya) ) ).

fof(a2_11,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n70)
        & accept_population(A,christian,n20)
        & accept_population(A,muslim,n8)
        & accept_population(A,native,n0)
        & accept_population(A,other,n2) )
    <=> accept_city(A,townb) ) ).

fof(a2_12,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n78)
        & accept_population(A,christian,n20)
        & accept_population(A,muslim,n1)
        & accept_population(A,native,n0)
        & accept_population(A,other,n1) )
    <=> accept_city(A,cityb) ) ).

fof(a2_13,axiom,
    ! [A] :
      ( ( accept_population(A,atheist,n30)
        & accept_population(A,christian,n0)
        & accept_population(A,muslim,n65)
        & accept_population(A,native,n0)
        & accept_population(A,other,n5) )
    <=> accept_city(A,townc) ) ).

%------------------------------------------------------------------------------

fof(conjecture, conjecture, ~(![Z]: (~(the_agent_not_in_any_proposed_teams(Z, a))))).