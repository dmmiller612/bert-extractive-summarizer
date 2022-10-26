import os
the_port = int(os.environ.get("PORT", 8080))

bind = "0.0.0.0:{}".format(the_port)
worker_class = 'gevent'
workers = int(os.environ.get("WORKER_THREADS", 1))
timeout = int(os.environ.get("WORKER_TIMEOUT", 720))
