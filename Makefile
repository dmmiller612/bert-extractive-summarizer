docker-service-build:
	docker build -t summary-service .

docker-service-run:
	docker run --rm -it -p 5011:8080 summary-service:latest -model bert-large-uncased
