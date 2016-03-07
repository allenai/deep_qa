(define sfe-spec-file (car ARGV))
(define graph-features (create-sfe-feature-computer sfe-spec-file))

(define word-family (word-graph-parameters)
  (lambda (word)
    (lambda (entity)
      (let ((var (make-entity-var entity))
            (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
        (make-featurized-classifier
          var (get-entity-features entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
        var))))

(define word-rel-family (word-rel-graph-parameters)
  (lambda (word)
    (lambda (entity1 entity2)
      (let ((var (make-entity-var (cons entity1 entity2)))
            (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
        (make-featurized-classifier
          var (get-entity-tuple-features entity1 entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
        var))))

(define get-word-cat (parameters)
  (let ((word-graph-parameters (get-ith-parameter parameters 0))
        (word-cat (word-family word-graph-parameters)))
    word-cat))

(define get-word-rel (parameters)
  (let ((word-rel-graph-parameters (get-ith-parameter parameters 1))
        (word-rel (word-rel-family word-rel-graph-parameters)))
    word-rel))

(define expression-family (parameters)
  (let ((word-cat (get-word-cat parameters))
        (word-rel (get-word-rel parameters)))
    (define expression-evaluator (expression entities)
      (eval expression))
    expression-evaluator))

(define word-ranking-family (word-graph-parameters)
  (lambda (word)
    (lambda (entity neg-entity)
      (let ((var (make-entity-var entity))
            (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
        (make-featurized-classifier
          var (get-entity-feature-difference entity neg-entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
        var))))

(define word-rel-ranking-family (word-rel-graph-parameters)
  (lambda (word)
    (lambda (entity1 neg-entity1 entity2 neg-entity2)
      (let ((var (make-entity-var (cons entity1 entity2)))
            (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
        (make-featurized-classifier
          var (get-entity-tuple-feature-difference entity1 entity2 neg-entity1 neg-entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
        var))))

(define get-ranking-word-cat (parameters)
  (let ((word-graph-parameters (get-ith-parameter parameters 0))
        (word-cat (word-ranking-family word-graph-parameters)))
    word-cat))

(define get-ranking-word-rel (parameters)
  (let ((word-rel-graph-parameters (get-ith-parameter parameters 1))
        (word-rel (word-rel-ranking-family word-rel-graph-parameters)))
    word-rel))

(define expression-ranking-family (parameters)
  (let ((word-cat (get-ranking-word-cat parameters))
        (word-rel (get-ranking-word-rel parameters))
        (expression-evaluator (get-expression-evaluator word-cat word-rel)))
    expression-evaluator))


(define expression-parameters
  (make-parameter-list (list (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-cat-word-feature-list x graph-features)))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-rel-word-feature-list x graph-features)))
                                                             (dictionary-to-array rel-words)))
                             )))

(define find-related-entities (midsInQuery midRelationsInQuery)
  (array-merge-sets (get-all-related-entities midsInQuery) (find-related-entities-in-graph midRelationsInQuery graph-features)))
