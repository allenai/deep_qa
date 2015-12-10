

(display "Running universal schema...")
(define output-model-filename (car ARGV))

(display "Making training data...")

(define training-data (array-zip training-inputs
                           (array-map (lambda (x) #t) training-inputs)))

(define random-cat-neg-example (cat-word-pred)
  (not (cat-word-pred (dictionary-rand-elt entities))))

(define weighted-random-cat-neg-example (cat-word-pred)
  (not (cat-word-pred (sample-histogram entity-histogram))))

(define random-rel-neg-example (rel-word-pred)
  (not (apply rel-word-pred (dictionary-rand-elt entity-tuples))))

(define weighted-random-rel-neg-example (rel-word-pred)
  (not (apply rel-word-pred (sample-histogram entity-tuple-histogram))))

(display "Made training data.")
(define expression-parameters (make-parameter-list (list 
                                                    (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array cat-words)))
                                                    (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array entities)))
                                                    (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array rel-words)))
                                                    (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality)) (dictionary-to-array entity-tuples)))
                                    )))
(perturb-parameters expression-parameters 0.1)

(display "Training...")
(define best-params (opt expression-family expression-parameters training-data))

(display "Serializing Parameters...")
(serialize best-params output-model-filename)

(define expression-eval (lambda (expr) ((expression-family best-params) expr entities)))