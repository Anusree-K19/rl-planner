(:action learned_bridge
 :parameters (?x ?y)
 :precondition (and (clear ?x) (clear ?y) (handempty) (ontable ?x) (ontable ?y))
 :effect (and (on ?x ?y) (not (clear ?y)) (not (ontable ?x))))
