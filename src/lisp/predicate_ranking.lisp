; UGLY, but it seems like the only way to have just one place where I specify the model, which is
; really orthogonal to how I train it.
(define define-expression-evaluator (quote
  (define expression-evaluator (expression entities word)
    (if (= 1 (length entities))
      ((eval expression) (car entities) (rejection-sample-histogram entity-histogram (array-get-ith-element cat-word-entities (get-index-with-unknown word cat-words))))
      (let ((rel-neg-example (rejection-sample-histogram entity-tuple-histogram (array-get-ith-element rel-word-entities (get-index-with-unknown word rel-words)))))
        ((eval expression) (car entities) (car rel-neg-example) (cadr entities) (cadr rel-neg-example)))
      ))
))
