(define best-params (deserialize (car ARGV)))

(define UNKNOWN-WORD "<UNK>")
(define BASELINE-PROB 0.5)
; original value
; (define BASELINE-PROB 0.9)

(define get-cluster (word word-cluster-dict word-cluster-names cluster-dict clusters)
  (if (dictionary-contains word word-cluster-dict)
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup word word-cluster-dict)) cluster-dict))
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup UNKNOWN-WORD word-cluster-dict)) cluster-dict))))

(define expression-eval (expr)
  (let ((word-parameters (get-ith-parameter best-params 0))
        (word-graph-parameters (get-ith-parameter best-params 1))
        (entity-parameters (get-ith-parameter best-params 2))
        (word-rel-parameters (get-ith-parameter best-params 3))
        (word-rel-graph-parameters (get-ith-parameter best-params 4))
        (entity-tuple-parameters (get-ith-parameter best-params 5))
        (us-word-cat (word-family word-parameters entity-parameters word-graph-parameters))
        (us-word-rel (word-rel-family word-rel-parameters entity-tuple-parameters word-rel-graph-parameters)))

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
