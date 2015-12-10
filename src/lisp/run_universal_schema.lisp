(define best-params (deserialize (car ARGV)))

(define expression-eval (lambda (expr) ((expression-family best-params) expr (dictionary-to-array entities))))