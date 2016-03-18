(define best-params (deserialize (caddr ARGV)))

(define BASELINE-PROB 0.5)
; original value
; (define BASELINE-PROB 0.9)

(define expression-eval (expr)
  (let ((us-word-cat (get-word-cat best-params))
        (us-word-rel (get-word-rel best-params)))

    (define word-cat (word)
      (lambda (entity)
        (let ((baseline-retval (dictionary-contains entity (get-cluster word cat-word-dict cat-word-cluster-names cat-cluster-dict cat-clusters)))
              (us-retval ((us-word-cat word) entity))
              (retval (amb true-false))
              (choose-baseline (amb (list #t #f) (list BASELINE-PROB (- 1.0 BASELINE-PROB))))
              )

          (require (or choose-baseline (= retval us-retval)))
          (require (or (not choose-baseline) (= retval baseline-retval)))
          retval
          )))

    (define word-rel (word)
      (lambda (entity1 entity2)
        (let ((baseline-retval (dictionary-contains (list entity1 entity2) (get-cluster word rel-word-dict rel-word-cluster-names rel-cluster-dict rel-clusters)))
              (us-retval ((us-word-rel word) entity1 entity2))
              (retval (amb true-false))
              (choose-baseline (amb (list #t #f) (list BASELINE-PROB (- 1.0 BASELINE-PROB))))
              )

          (require (or choose-baseline (= retval us-retval)))
          (require (or (not choose-baseline) (= retval baseline-retval)))
          retval
          )))

    (define entities (dictionary-to-array entities))

    (eval expr)))






