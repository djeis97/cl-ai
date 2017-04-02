;;;; cl-ai.lisp

(in-package #:cl-ai)
(named-readtables:in-readtable rutilsx-readtable)

;;; "cl-ai" goes here. Hacks and glory await!

(defpar *training-factor* 0.001)
(defpar *alpha* 0.0001)

(defclass gate ()
  ((last-output :accessor gate-last-output)
   (last-gradient :accessor gate-last-gradient)
   (input-spec :accessor gate-input-spec :initarg :input-spec)
   (output-spec :accessor gate-output-spec :initarg :output-spec))) ;; ABS for gates
(defgeneric process (gate inp))
(defgeneric train (gate output-gradient))


(defclass multiply-gate (gate)
  ()
  (:default-initargs
   :input-spec '(number number)
   :output-spec '(number)))

(defmethod process ((gate multiply-gate) inp)
  (bind (((x y) inp)
         (output (* x y))
         (gradient (list y x)))
    (:= (? gate 'last-gradient) gradient
        (? gate 'last-output) output)
    output))
(defmethod train ((gate multiply-gate) output-gradient))

(defclass add-gate (gate)
  ()
  (:default-initargs
   :input-spec '(number number)
   :output-spec '(number)))
(defmethod process ((gate add-gate) inp)
  (bind (((x y) inp)
         (output (+ x y))
         (gradient (list 1 1)))
    (:= (? gate :last-gradient) gradient
        (? gate :last-output) output)
    output))
(defmethod train ((gate add-gate) output-gradient))

(defclass add-mul-circuit (gate)
  ((add-gate :initarg :add-gate)
   (multiply-gate :initarg :multiply-gate))
  (:default-initargs
   :input-spec '(number number number)
   :output-spec '(number)
   :add-gate (make-instance 'add-gate)
   :multiply-gate (make-instance 'multiply-gate)))

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
(defmethod train ((gate add-mul-circuit) output-gradient))

(declaim (inline vector-dot-product))
(defun vector-dot-product (a b)
  (declare (vector a b)
           (optimize (speed 3)))
  (loop :for a-val :being :the :elements :of a
        :for b-val :being :the :elements :of b
        :sum (* a-val b-val)))

(defclass support-vector-machine (gate)
  ((weights :initarg :weights)
   (bias :initarg :bias)
   weights-gradient)
  (:default-initargs
   :weights (vector 0 0)
   :bias 0
   :input-spec '((simple-vector 2))
   :output-spec '(number)))
(defmethod process ((gate support-vector-machine) inp)
  (bind ((outp (+ (vector-dot-product (car inp)
                                      (? gate :weights))
                   (? gate :bias))))
    (:= (? gate :last-gradient) (? gate :weights)
        (? gate :weights-gradient) (car inp)
        (? gate :last-output) outp)
    outp))
(defmethod train ((gate support-vector-machine) output-gradient)
  (bind ((grad (car output-gradient))
         (bias (? gate :bias))
         (weights (? gate :weights))
         (weights-gradient (cl-ana.gmath:* *training-factor*
                                           grad
                                           (? gate :weights-gradient)))
         (normalization (cl-ana.gmath:* *alpha* -1 weights)))
    (:= (? gate :weights) (cl-ana.gmath:+
                              weights
                              weights-gradient
                              normalization)
        (? gate :bias) (+ bias (* *training-factor* grad)))
    (cl-ana.gmath:* grad (? gate :last-gradient))))


(defun train-gate (gate data)
  (reduce #'cl-ana.gmath:+ (mapcar #`(bind ((inp (car %))
                                            (exp (cdr %))
                                            (outp (process gate inp))
                                            (err (cl-ana.gmath:- exp outp)))
                                       (train gate err)
                                       (cl-ana.tensor:tensor-map #'abs err))
                                   data)))
(defun test-gate (gate data)
  (values (mapcar #`(process gate (car %))
                  data)
          (mapcar #'second data)))
