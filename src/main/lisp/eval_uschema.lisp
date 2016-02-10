(define best-params (deserialize (cadr ARGV)))

(define expression-eval (lambda (expr) ((expression-family best-params) expr (dictionary-to-array entities))))
