;; Requires the user to define the following variables 
;; in a data file before running this program:
;; entities -- e.g. (define entities (list "/en/plano" "/en/Texas" "/en/banana"))
;; cat-words    -- e.g. (define words (list "city" "place" "fruit" "in"))
;; training-inputs -- e.g., 
;;   (define training-inputs
;;     (list (list (quote (exists-func (lambda (x) (and ((mention "/en/plano") x) ((word-cat "city") x))) cur-entities)) entities)
;;           (list (quote (exists-func (lambda (y) (and ((mention "/en/Texas") y) 
;;                                                (exists-func (lambda (x) (and ((mention "/en/plano") x) ((word-cat "city") x)) ((word-rel "in") x y)) cur-entities))) 
;;                               cur-entities)) entities)
;;           ))

(define true-false (list #t #f))
(define latent-dimensionality 300)

(define UNKNOWN-WORD "<UNK>")

(define entity-array (dictionary-to-array entities))
(define entity-tuple-array (dictionary-to-array entity-tuples))

;; Constant functions ;;;;;;;;;;;;;;;;;;;;;;

;; Returns the set containing only entity-name
(define mention (entity-name)
  (lambda (x) (= entity-name x)))

;; Computes the set of entities contained in a predicate
;; (represented as a function)
(define predicate-to-set (predicate candidate-set)
  (array-map (lambda (entity) (predicate entity)) candidate-set))

;; Returns true if any element of set1 is true.
(define any-is-true? (set1) 
  (array-foldr or set1 #f))

;; Returns true if func returns true for any candidate
;; in candidate-set
(define exists-func (func candidate-set)
  (any-is-true? (predicate-to-set func candidate-set)))

;; Returns the intersection of two sets of entities. Each set
;; is represented by a list of binary indicator variables for
;; the presence of each entity.
(define intersect (set1 set2)
  (lifted-map (lambda (x) (and (lifted-car x) (lifted-car (lifted-cdr x)))) (lifted-zip set1 set2)))

;; Returns true if any element in set1 is also present in set2.
;; Equivalent to \exists x. x \in set1 ^ x \in set2
(define contains-any? (set1 set2)
  (lifted-foldr or (intersect set1 set2) #f))

(define => (x y)
  (or (not x) y))

;; Returns true if all elements of set1 are present in set2.
;; Equivalent to set1 \subset set2
(define contains-all? (set1 set2)
  (lifted-foldr 
   and
   (lifted-map (lambda (tuple) (=> (lifted-car tuple) (lifted-cadr tuple)))
               (lifted-zip set1 set2))
   #t))

(define get-elementwise-marginals (predicate-amb-list)
  (lifted-map (lambda (x) (get-marginals x)) predicate-amb-list))

(define get-predicate-marginals (predicate candidate-array)
  (array-map (lambda (x) (let ((marginals (get-marginals (predicate x)))
			       (ind (find-index #t (car marginals))))
			   (if (= ind -1)
			       (list 0.0 x)
			       (list (get-ith-element (cadr marginals) ind) x ))))
	     candidate-array))

(define print-predicate-marginals (predicate candidate-array)
  (begin 
    (array-map (lambda (x) (new-fg-scope (let ((marginals (get-marginals (predicate x)))
					       (ind (find-index #t (car marginals))))
					   (if (= ind -1)
;(display 0.0 x)
					       #t 
					       (let ((prob (get-ith-element (cadr marginals) ind)))
						 (if (= prob 0.0)
						     #t
						   (display prob x )))))))
	     candidate-array) 
    #t))

(define print-relation-marginals (predicate candidate-array)
  (begin 
    (array-map (lambda (x) (let ((marginals (get-marginals (predicate (car x) (cadr x))))
				 (ind (find-index #t (car marginals))))
			     (if (= ind -1)
				 (display 0.0 x)
				 (display (get-ith-element (cadr marginals) ind) x ))))
	     candidate-array)
    #t))


(define get-related-entities (entity)
  (if (dictionary-contains entity entities)
      (array-get-ith-element related-entities (dictionary-lookup entity entities))
      (array )
      )
)

(define get-all-related-entities (entity-list)
  (if (= 0 (length entity-list))
      entity-array
      (foldr array-merge-sets (map get-related-entities entity-list) (array ))))

;; Trainable functions ;;;;;;;;;;;;;;;;;;;;;;;

;; Creates a true/false variable for entity-name
(define make-entity-var (entity-name)
  (amb true-false))

;; Functions for looking up parameters corresponding to
;; a particular word / entity.
(define get-params (element element-list parameter-list unknown-elt)
  (if (dictionary-contains element element-list)
      (get-ith-parameter parameter-list (dictionary-lookup element element-list))
      (get-ith-parameter parameter-list (dictionary-lookup unknown-elt element-list))))

(define get-index-with-unknown (word word-dict)
  (if (dictionary-contains word word-dict)
      (dictionary-lookup word word-dict)
      (dictionary-lookup UNKNOWN-WORD word-dict)))

(define get-cat-word-params (word word-parameter-list)
  (get-params word cat-words word-parameter-list UNKNOWN-WORD))

(define get-rel-word-params (word word-parameter-list)
  (get-params word rel-words word-parameter-list UNKNOWN-WORD))

(define get-entity-params (entity entity-parameter-list)
  (get-params entity entities entity-parameter-list #f))

(define get-entity-tuple-params (entity-arg1 entity-arg2 entity-tuple-parameter-list)
  (get-params (list entity-arg1 entity-arg2) entity-tuples entity-tuple-parameter-list #f))

(define word-family (word-parameters entity-parameters)
  (lambda (word)
    (lambda (entity)
      (if (dictionary-contains entity entities)
	  (let ((var (make-entity-var entity)))
	    (make-inner-product-classifier 
	     var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters))
	    var)
	  #f)
      )))

(define word-rel-family (word-rel-params entity-tuple-params)
  (define word-rel (word)
    (lambda (entity1 entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
	  (let ((var (make-entity-var (cons entity1 entity2))))
	    (make-inner-product-classifier 
	     var #t (get-rel-word-params word word-rel-params)
	     (get-entity-tuple-params entity1 entity2 entity-tuple-params))
	    var)
	  #f
	  )
      ))
  word-rel)

(define expression-family (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (entity-parameters (get-ith-parameter parameters 1))
        (word-rel-parameters (get-ith-parameter parameters 2))
        (entity-tuple-parameters (get-ith-parameter parameters 3))
        (word-cat (word-family word-parameters entity-parameters))
        (word-rel (word-rel-family word-rel-parameters entity-tuple-parameters)))
    (define expression-evaluator (expression entities)
      (eval expression))
    expression-evaluator))

(define print-parameters (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (entity-parameters (get-ith-parameter parameters 1))
        (word-rel-parameters (get-ith-parameter parameters 2))
        (entity-tuple-parameters (get-ith-parameter parameters 3)))
; (array-map (lambda (word) (display word (parameters-to-string (get-cat-word-params word word-parameters))))
;	       (dictionary-to-array cat-words))

    (array-map (lambda (word) (display word (parameters-to-string (get-rel-word-params word word-rel-parameters))))
	       (dictionary-to-array rel-words))
))

;; (display "exists x st. (plano x) and (city x)?")
;; (display (get-marginals (expression-eval (quote (exists-func (lambda (x) (and ((mention "/en/plano") x) ((word-cat "city") x))) entities)))))
;; 
;; (display "exists x,y st. (plano x) and (city x) and (texas y) and (in x y)?")
;; (display (get-marginals (expression-eval (quote (exists-func (lambda (y) (and ((mention "/en/Texas") y) 
;;                                              (exists-func (lambda (x) (and ((mention "/en/plano") x) ((word-cat "city") x)) ((word-rel "in") x y)) entities))) 
;;                             entities)))))
;; 
;; 
;; (display "exists x st. (city x)")
;; (display (get-marginals (expression-eval (quote (and (exists-func (word-cat "city") entities))))))
;; 
;; 
;; (display "exists x st. (city x) and exists y st. (city y)")
;; (display (get-marginals (expression-eval (quote (and (exists-func (word-cat "city") entities)
;;                                                      (exists-func (word-cat "city") entities) )))))
;; 
;; 
;; (display "lambda x. st (city x) (in x y) (texas y)")
;; (display entities)
;; (display (get-best-value (expression-eval (quote (predicate-to-set (lambda (x) (and ((word-cat "city") x)
;;                                                                                    (exists-func (lambda (y) (and ((word-rel "in") x y) ((mention "/en/Texas") y))) entities))) entities)))))
