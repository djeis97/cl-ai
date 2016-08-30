;;;; cl-ai.asd

(asdf:defsystem #:cl-ai
  :description "Describe cl-ai here"
  :author "Elijah Malaby"
  :license "Specify license here"
  :depends-on (#:cl-ana #:alexandria #:rutils #:rutilsx)
  :serial t
  :components ((:file "package")
               (:file "cl-ai")))

