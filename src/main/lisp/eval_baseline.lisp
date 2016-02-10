(define expression-eval (expr)
  (define word-cat (word)
    (lambda (entity)
      (let ((retval (dictionary-contains entity (get-cluster word cat-word-dict cat-word-cluster-names cat-cluster-dict cat-clusters))))
        retval
        )
      ))

  (define word-rel (word)
    (lambda (entity1 entity2)
      (dictionary-contains (list entity1 entity2) (get-cluster word rel-word-dict rel-word-cluster-names rel-cluster-dict rel-clusters))
      )
    )

  (define entities (dictionary-to-array entities))
  (eval expr))
