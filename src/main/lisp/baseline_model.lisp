(define get-cluster (word word-cluster-dict word-cluster-names cluster-dict clusters)
  (if (dictionary-contains word word-cluster-dict)
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup word word-cluster-dict)) cluster-dict))
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup UNKNOWN-WORD word-cluster-dict)) cluster-dict))))
