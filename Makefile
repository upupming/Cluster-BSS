PY = python

all: dg cluster

dg:
	${PY} data_generator.py

cluster:
	${PY} cluster.py 

clean:
	rm ./figures/* ./results/*