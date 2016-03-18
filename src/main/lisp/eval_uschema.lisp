(define best-params (deserialize (caddr ARGV)))

(define expression-eval (lambda (expr) ((expression-family best-params) expr (dictionary-to-array entities))))
