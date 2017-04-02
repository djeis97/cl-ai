
(eval-when (:compile-toplevel :load-toplevel)
  (ql:quickload :rutilsx)
  (ql:quickload :cl-ana))

(use-package :rutilsx)
(named-readtables:in-readtable rutilsx-readtable)
(cl-ana.gmath:use-gmath *package*)
(use-package :cl-ana)

(defparameter *x* (variable 'integer))
(defparameter *y* (variable 'integer))

(defparameter *input* (variable '(array float (80 80 3))))
(defparameter *kernel1* (variable '(array float (7 7 3 20))))
(defparameter *kernel2* (variable '(array float (7 7 20 20))))
(defparameter *h1* (variable '(array float (5120 100))))
(defparameter *h2* (variable '(array float (100 5))))
(defparameter *classifications* (list :a :b :c :d :e))

#-(and)
(defparameter *model* (-> *x*
                          (convolve *kernel1*)
                          (downsample '(2 2))
                          (convolve *kernel2*)
                          (downsample '(2 2))
                          (* *h1*)
                          (* *h2*)))
#-(and)
(defparameter *error* (- *model* *y*))

(run *model* (env *x* image))

#-(and)
(classify *classifications* (gethash *model* (run *model* (env *x* image))))

#-(and)
(gradient (* *x* 2) *x*)

#-(and)
(optimize *error*
          (dataset *x* images *y* classifications)
          (list *kernel1* *kernel2* *h1* *h2*))

(defclass graph-node () ())

(defun env (&rest env-alist)
  (loop
    :with h := (make-hash-table)
    :for (a b) :on env-alist :by #'cddr :do
      (assert (typep b (output-type a)))
      (setf (gethash a h) b)
    :finally (return h)))

(defun run (graph-node environment &optional (cache (make-hash-table :test #'equal)))
  "Runs calculations to compute value of graph-node. Computations will be cached in cache, which will be returned as second value."
  (values (run-internal graph-node environment cache) cache))

(defgeneric run-internal (graph-node environment cache)
  (:documentation "Runs calculations to compute value of graph-node. Computations will be cached in cache, which will be returned as second value."))

(defmethod run-internal :around ((graph-node graph-node) environment cache)
  (aif (gethash graph-node cache)
       it
       (setf (gethash graph-node cache) (call-next-method))))

(defmethod run-internal (val environment cache)
  (declare (ignore environment cache))
  val)

(defmethod run-internal ((graph-node graph-node) environment cache)
  (declare (ignore cache))
  (or (gethash graph-node environment) (call-next-method)))

(defgeneric output-type (graph-node)
  (:documentation "Type specifier of result from running graph-node"))

(defmethod output-type (thingy)
  (type-of thingy))

(defun gradient (a b env &optional (cache (make-hash-table :test #'equal)))
  "Calculate the gradient of graph-node a with respect to graph-node b in env."
  (values (gradient-internal a b env cache) cache))

(defgeneric gradient-internal (a b env cache)
  (:documentation "Calculate the gradient of graph-node a with respect to graph-node b in env."))

(defmethod gradient-internal :around ((a graph-node) (b graph-node) env cache)
  (let ((k (cons a b)))
    (acond
      ((eq a b) 1)
      ((gethash k cache) it)
      (t (setf (gethash k cache) (call-next-method))))))

(defmethod gradient-internal (a b env cache) 0)

(defclass graph-variable (graph-node)
  ((output-type :initarg :output-type :reader output-type)))

(defun variable (type)
  (make-instance 'graph-variable :output-type type))

(defmethod gradient-internal ((v graph-variable) dv env cache)
  (declare (ignore v dv env cache))
  0)


(defclass simple-math-node (graph-node)
  ((inputs :initarg :inputs)
   (output-type :reader output-type)))

(defun parent-type (&optional (a t) (b t))
  (cond
    ((subtypep a b) b)
    ((subtypep b a) a)
    (t (error "Incompatible types: ~A ~A" a b))))

(defmethod initialize-instance :after ((m simple-math-node) &key inputs &allow-other-keys)
  (setf (slot-value m 'output-type) (reduce #'parent-type inputs :key #'output-type)))

(defclass addition (simple-math-node) ())

(defmethod run-internal ((a addition) env cache)
  (reduce #'add (slot-value a 'inputs) :key (lambda (i) (run-internal i env cache))))

(defmethod-commutative add ((x graph-node) y)
  (make-instance 'addition :inputs (list x y)))
(defmethod-commutative add ((x addition) y)
  (make-instance 'addition :inputs (cons y (slot-value x 'inputs))))

(defmethod gradient-internal ((a addition) b env cache)
  (reduce #'add (slot-value a 'inputs) :key (lambda (i) (gradient-internal i b env cache))))

(defclass multiplication (simple-math-node) ())

(defmethod run-internal ((m multiplication) env cache)
  (reduce #'mult (slot-value m 'inputs) :key (lambda (i) (run-internal i env cache))))

(defmethod gradient-internal ((a multiplication) b env cache)
  (if (member b (slot-value a 'inputs))
      (reduce #'mult (remove b (slot-value a 'inputs)) :key (lambda (i) (run-internal i env cache)))
      0))

(defmethod-commutative mult ((x graph-node) y)
  (make-instance 'multiplication :inputs (list x y)))
(defmethod-commutative mult ((x multiplication) y)
  (make-instance 'multiplication :inputs (cons y (slot-value x 'inputs))))


(defclass subtraction (simple-math-node) ())

(defmethod run-internal ((s subtraction) env cache)
  (let ((inputs (slot-value s 'inputs)))
    (if (cdr inputs)
        (reduce #'sub inputs :key (lambda (i) (run-internal i env cache)))
        (unary-sub (run-internal (car inputs) env cache)))))

(defmethod gradient-internal ((s subtraction) b env cache)
  (reduce #'sub (slot-value :key (lambda (i) (gradient-internal i b env cache)))))

(defmethod sub ((x graph-node) y)
  (make-instance 'subtraction :inputs (list x y)))
(defmethod sub (x (y graph-node))
  (make-instance 'subtraction :inputs (list x y)))
(defmethod sub ((x subtraction) y)
  (make-instance 'subtraction :inputs (append (slot-value x 'inputs) (list y))))

(defmethod unary-sub ((x graph-node))
  (make-instance 'subtraction :inputs (list x)))
(defmethod unary-sub ((x subtraction))
  (make-instance 'addition :inputs (slot-value x 'inputs)))

(defclass division (simple-math-node) ())

(defmethod run-internal ((d division) env cache)
  (let ((inputs (slot-value d 'inputs)))
    (if (cdr inputs)
        (reduce #'div inputs :key (lambda (i) (run-internal i env cache)))
        (unary-div (run-internal (car inputs) env cache)))))

(defmethod gradient-internal ((a division) b env cache)
  (reduce #'
   ))

;; (+ (* (run a) (gradient v b)) (* (run b) (gradient v a)))
