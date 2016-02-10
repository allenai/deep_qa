;(display "Deserializing the model")
;(define best-params (deserialize (car ARGV)))
;
(define best-params
  (make-parameter-list (list (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-cat-word-feature-list x)))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entities)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array rel-words)))
                             (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-rel-word-feature-list x)))
                                                             (dictionary-to-array rel-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entity-tuples)))
                             )))

(define expression-eval (lambda (expr) ((expression-family best-params) expr (dictionary-to-array entities))))

(define attorney-index (dictionary-lookup "attorney" cat-words))
(define cat-word-params (get-ith-parameter best-params 0))
(define cat-word-graph-params (get-ith-parameter best-params 1))

;;(display "cat-word-params for attorney")
;;(display-parameters (get-params "attorney" cat-words cat-word-params UNKNOWN-WORD))

;;(display "cat-word-graph-params for attorney")
;;(display-parameters (get-params "attorney" cat-words cat-word-graph-params UNKNOWN-WORD))

(display
  (expression-eval (quote (get-marginals
                            (and ((word-cat "supermarket") "/m/029jpy") )
                            ))))
