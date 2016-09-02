;;;; cl-ai.lisp

(in-package #:cl-ai)

;;; "cl-ai" goes here. Hacks and glory await!

(defsetf smart-slot-value (object slot-name) (val)
  (alexandria:with-gensyms (slot-sym)
    (alexandria:once-only (object slot-name)
      `(let ((,slot-sym (or (find-symbol (string-upcase ,slot-name)
                                    (symbol-package (class-name (class-of ,object))))
                       ,slot-name)))
         (setf (slot-value ,object ,slot-sym) ,val)))))

(defclass gate ()
  ((last-output :accessor gate-last-output)
   (last-gradient :accessor gate-last-gradient)
   (input-dims :accessor gate-input-dims)
   (output-dims :accessor gate-output-dims))) ;; ABS for gates
(defgeneric process (gate inp))
(defgeneric train (gate output-gradient))


(defclass multiply-gate (gate)
  ((input-dims :initform '(2))
   (output-dims :initform '(1))))
(defmethod process ((gate multiply-gate) inp)
  (bind (((x y) inp)
         (output (* x y))
         (gradient (list y x)))
    (:= (? gate 'last-gradient) gradient
        (? gate 'last-output) output)
    output))
(defmethod train ((gate multiply-gate) output-gradient))

(defclass add-gate (gate)
  ((input-dims :initform '(2))
   (output-dims :initform '(1))))
(defmethod process ((gate add-gate) inp)
  (bind (((x y) inp)
         (output (+ x y))
         (gradient (list 1 1)))
    (:= (? gate :last-gradient) gradient
        (? gate :last-output) output)
    output))
(defmethod train ((gate add-gate) output-gradient))

(defclass add-mul-circuit (gate)
  ((input-dims :initform '(3))
   (output-dims :initform '(1))
   (add-gate :initform (make-instance 'add-gate))
   (multiply-gate :initform (make-instance 'multiply-gate))))
(defmethod process ((gate add-mul-circuit) inp)
  (bind (((x y z) inp)
         (added (process (? gate :add-gate) (list x y)))
         (outp (process (? gate :multiply-gate) (list added z)))
         ((deriv-outp-wrt-added deriv-outp-wrt-z)
          (? gate :multiply-gate :last-gradient))
         ((deriv-added-wrt-x deriv-added-wrt-y)
          (? gate :add-gate :last-gradient))
         (deriv-outp-wrt-x (* deriv-outp-wrt-added deriv-added-wrt-x))
         (deriv-outp-wrt-y (* deriv-outp-wrt-added deriv-added-wrt-y)))
    (:= (? gate :last-gradient) (list deriv-outp-wrt-x
                                      deriv-outp-wrt-y
                                      deriv-outp-wrt-z)
        (? gate :last-output) outp)
    outp))

