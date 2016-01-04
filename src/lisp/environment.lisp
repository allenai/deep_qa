;; list functions

(define nil (list))

;; Apply function f to every element of seq
(define map (lambda (f seq) (begin (if (nil? seq) (list) (cons (f (car seq)) (map f (cdr seq)))))))
(define lifted-map (lambda (f seq) (if (nil? seq) (lifted-list) (lifted-cons (f (lifted-car seq)) (lifted-map f (lifted-cdr seq))))))

;; Combines the elements of seq using f.
(define foldr (lambda (f seq init)
                (if (nil? seq)
                  init
                  (f (car seq) (foldr f (cdr seq) init)))))

(define lifted-foldr (lambda (f seq init)
                       (if (nil? seq)
                         init
                         (f (lifted-car seq) (lifted-foldr f (lifted-cdr seq) init)))))

(define zip (list1 list2) (if (nil? list1) (list) (cons (list (car list1) (car list2)) (zip (cdr list1) (cdr list2)))))
(define lifted-zip (list1 list2) (if (nil? list1) (lifted-list) (lifted-cons (lifted-list (lifted-car list1) (lifted-car list2)) (lifted-zip (lifted-cdr list1) (lifted-cdr list2)))))

(define zip3 (list1 list2 list3) (if (nil? list1) (list) (cons (list (car list1) (car list2) (car list3)) (zip3 (cdr list1) (cdr list2) (cdr list3)))))

(define cadr (x) (car (cdr x)))
(define caddr (x) (car (cdr (cdr x))))
(define cadddr (x) (car (cdr (cdr (cdr x)))))

(define lifted-cadr (x) (lifted-car (lifted-cdr x)))
(define lifted-caddr (x) (lifted-car (lifted-cdr (lifted-cdr x))))
(define lifted-cadddr (x) (lifted-car (lifted-cdr (lifted-cdr (lifted-cdr x)))))

(define length (lambda (seq) (if (nil? seq) 0 (+ (length (cdr seq)) 1))))
(define first-n (lambda (seq n) (if (= n 0) (list) (cons (car seq) (first-n (cdr seq) (- n 1))))))
(define remainder-n (lambda (seq n) (if (= n 0) seq (remainder-n (cdr seq) (- n 1)))))
(define get-ith-element (lambda (seq i) (if (nil? seq) (list)
                                          (if (= i 0) (car seq)
                                            (get-ith-element (cdr seq) (- i 1))))))

(define n-to-1 (lambda (n) (if (= n 0) (list) (cons n (n-to-1 (- n 1))))))

(define find-index (elt seq) (find-index-helper elt seq 0))
(define find-index-helper (elt seq i)
  (if (nil? seq)
    -1
    (if (= (car seq) elt)
      i
      (find-index-helper elt (cdr seq) (+ i 1)))))

(define lifted-get-ith-element (lambda (seq i) (if (nil? seq) (lifted-list)
                                                 (if (= i 0) (lifted-car seq)
                                                   (lifted-get-ith-element (lifted-cdr seq) (- i 1))))))

(define lifted-cadr (x) (lifted-car (lifted-cdr x)))

(define lifted-alist-find (elt alist)
  (if (nil? alist)
    (lifted-list)
    (if (= (lifted-car (lifted-car alist)) elt)
      (lifted-car (lifted-cdr (lifted-car alist)))
      (lifted-alist-find elt (lifted-cdr alist)))))

(define lifted-list-eq? (l1 l2)
  (if (or (nil? l1) (nil? l2))
    (and (nil? l1) (nil? l2))
    (and (= (lifted-car l1) (lifted-car l2))
         (lifted-list-eq? (cdr l1) (cdr l2)))))

;; Appends two lists.
(define append (lambda (l1 l2)
                 (if (nil? l1)
                   l2
                   (cons (car l1) (append (cdr l1) l2)))))

(define flatten (lambda (list-of-lists)
                  (if (nil? list-of-lists)
                    (list)
                    (append (car list-of-lists) (flatten (cdr list-of-lists))))))

;; Takes a list of lists and returns a list of tuples of every possible
;; pair (triplet, etc.) of elements.
(define outer-product (list-of-lists)
  (if (nil? list-of-lists)
    (list (list))
    (outer-product-helper (car list-of-lists) (outer-product (cdr list-of-lists)))))

(define outer-product-helper (input rest)
  (if (nil? input)
    (list)
    (append (map (lambda (x) (cons (car input) x)) rest)
            (outer-product-helper (cdr input) rest))))

;; Generates a list containing the elements n (n-1) ... 1
(define 1-to-n (lambda (n) (if (= n 0) (list) (cons n (1-to-n (- n 1))))))

(define generate-int-seq (start end)
  (if (or (> start end) (= start end))
    (list)
    (cons start (generate-int-seq (+ start 1) end))))

;; Forces all nondeterministic executions to satisfy the given condition
(define require (lambda (condition) (add-weight (not condition) 0.0)))

;; Math functions
(define square (lambda (n) (* n n)))

;; Training data stuff
(define make-eq-require (output-value) (lambda (x) (require (= x output-value))))
