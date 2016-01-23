; UGLY, but it seems like the only way to have just one place where I specify the model, which is
; really orthogonal to how I train it.
(define define-expression-evaluator (quote
  (define expression-evaluator (expression arg-entities arg1s arg2s positive-set-ind)
    (let ((arg1-candidates (apply dset-intersect (map (lambda (x) (array-get-ith-element arg1-arg2-map (dictionary-lookup x entities))) arg1s)))
          (arg2-candidates (apply dset-intersect (map (lambda (x) (array-get-ith-element arg2-arg1-map (dictionary-lookup x entities))) arg2s)))
          (all-candidates (dset-intersect arg1-candidates arg2-candidates))
          (positive-examples (array-get-ith-element joint-entities positive-set-ind)))

      (if (nil? all-candidates)
        ;; No constraints on the argument value.
        (begin
          ((eval expression) (car arg-entities) (rejection-sample-histogram entity-histogram positive-examples))
          )

        (let ((new-candidates (dset-subtract all-candidates positive-examples)))
          (if (not (dset-empty? new-candidates))
            ((eval expression) (car arg-entities) (sample-histogram-conditional entity-histogram new-candidates))
            ; This #f results in a search error. The search error happens because
            ; every possible value for the sampled entity has occurred as an answer to this query.
            #f
            )
          )
        )
      ))
))
