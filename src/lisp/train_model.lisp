(display "Running universal schema")
(define output-model-filename (cadr ARGV))

(display "Making training data")
(define training-data (array-zip training-inputs
                                 (array-map (lambda (x) #t) training-inputs)))

(display "Perturbing parameters")
(perturb-parameters expression-parameters 0.1)

(display "Training")
(define best-params (opt expression-ranking-family expression-parameters training-data))

(display "Serializing Parameters")
(serialize best-params output-model-filename)
