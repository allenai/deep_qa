(display "Running universal schema...")
(define output-model-filename (car ARGV))

(display "Making training data...")

(define training-data (array-zip training-inputs
                                 (array-map (lambda (x) #t) training-inputs)))

; (define random-cat-neg-example (cat-word-pred)
;   (not (cat-word-pred (dictionary-rand-elt entities))))
;
; (define weighted-random-cat-neg-example (cat-word-pred)
;   (not (cat-word-pred (sample-histogram entity-histogram))))
;
; (define random-rel-neg-example (rel-word-pred)
;   (not (apply rel-word-pred (dictionary-rand-elt entity-tuples))))
;
; (define weighted-random-rel-neg-example (rel-word-pred)
;   (not (apply rel-word-pred (sample-histogram entity-tuple-histogram))))


(define word-ranking-family (word-parameters entity-parameters)
  (lambda (word)
    (lambda (entity neg-entity)
      (if (dictionary-contains entity entities)
        (let ((var (make-entity-var entity)))
          (make-ranking-inner-product-classifier
            var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters)
            (get-entity-params neg-entity entity-parameters))
          (make-featurized-classifier
            var (get-entity-feature-difference entity neg-entity) (get-cat-word-params word word-graph-parameters))
          var)
        #f)
      )))

(define word-rel-ranking-family (word-rel-params entity-tuple-params)
  (define word-rel (word)
    (lambda (entity1 neg-entity1 entity2 neg-entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
        (let ((var (make-entity-var (cons entity1 entity2))))
          (make-ranking-inner-product-classifier
            var #t (get-rel-word-params word word-rel-params)
            (get-entity-tuple-params entity1 entity2 entity-tuple-params)
            (get-entity-tuple-params neg-entity1 neg-entity2 entity-tuple-params))
          (make-featurized-classifier
            var (get-entity-tuple-feature-difference entity1 entity2 neg-entity1 neg-entity2) (get-rel-word-params word word-rel-graph-parameters))
          var)
        #f
        )
      ))
  word-rel)

(define expression-ranking-family (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (word-graph-parameters (get-ith-parameter parameters 1))
        (entity-parameters (get-ith-parameter parameters 2))
        (word-rel-parameters (get-ith-parameter parameters 3))
        (word-rel-graph-parameters (get-ith-parameter parameters 4))
        (entity-tuple-parameters (get-ith-parameter parameters 5))
        (word-cat (word-ranking-family word-parameters entity-parameters word-graph-parameters))
        (word-rel (word-rel-ranking-family word-rel-parameters entity-tuple-parameters word-rel-graph-parameters)))
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
    expression-evaluator))


(display "Made training data.")
(define expression-parameters (make-parameter-list (list
                                                     (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array cat-words)))
                                                     ;; I'm not sure at all about the (list (list #t #f)) part here...
                                                     (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list (list #t #f)) word-graph-features)) (dictionary-to-array cat-words)))
                                                     (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array entities)))
                                                     (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array rel-words)))
                                                     ;; I'm not sure at all about the (list (list #t #f)) part here...
                                                     (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list (list #t #f)) word-rel-graph-features)) (dictionary-to-array rel-words)))
                                                     (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array entity-tuples)))
                                                     )))
(perturb-parameters expression-parameters 0.1)

(display "Training...")
(define best-params (opt expression-ranking-family expression-parameters training-data))

(display "Serializing Parameters...")
(serialize best-params output-model-filename)

(define expression-eval (lambda (expr) ((expression-family best-params) expr entities)))
