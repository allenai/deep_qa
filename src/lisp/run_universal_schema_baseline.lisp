;; Requires the user to define the following variables:
;; Dictionary from words to array indexes in word-cluster-names
;; cat-word-dict rel-word-dict
;;
;; Array with the name of the cluster for each index in word-dict
;; cat-word-cluster-names rel-word-cluster-names
;;
;; Map from cluster names to indexes in clusters
;; cat-cluster-dict rel-cluster-dict
;;
;; Array containing dictionaries for the entities in each cluster
;; cat-clusters rel-clusters

(define UNKNOWN-WORD "<UNK>")

(define get-cluster (word word-cluster-dict word-cluster-names cluster-dict clusters)
  (if (dictionary-contains word word-cluster-dict)
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup word word-cluster-dict)) cluster-dict))
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup UNKNOWN-WORD word-cluster-dict)) cluster-dict))))

(define expression-eval (expr)
  (define word-cat (word)
    (lambda (entity)
      (let ((retval (dictionary-contains entity (get-cluster word cat-word-dict cat-word-cluster-names cat-cluster-dict cat-clusters))))
        retval
        )
      ))

  (define word-rel (word)
    (lambda (entity1 entity2)
      (dictionary-contains (list entity1 entity2) (get-cluster word rel-word-dict rel-word-cluster-names rel-cluster-dict rel-clusters))
      )
    )

  (define entities (dictionary-to-array entities))
  (eval expr))
