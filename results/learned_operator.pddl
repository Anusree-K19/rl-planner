(:action learned_bridge
 :parameters (?x ?y)
 :precondition (and (clear ?x) (clear ?y) (handempty) (ontable ?x))
 :effect (and (on ?x ?y) (not (clear ?y)) (not (ontable ?x))))
