IMAGE_NAME = enveda/spec2tax
AWS_ACCESS_KEY_ID := $(shell aws --profile default configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY := $(shell aws --profile default configure get aws_secret_access_key)
AWS_DEFAULT_REGION := us-east-1
AWS_IMAGE_NAME := <put your image name here>

build:
	docker build -t $(IMAGE_NAME) .

push:
	aws ecr get-login-password --region $(AWS_DEFAULT_REGION) | docker login --username AWS --password-stdin $(AWS_IMAGE_NAME)
	docker build -t $(IMAGE_NAME) .
	docker tag $(IMAGE_NAME) $(AWS_IMAGE_NAME)
	docker push $(AWS_IMAGE_NAME)

submit-job-mammalia:
	aws sagemaker create-processing-job \
		--processing-job-name spec2fam-$$(date +%s)
		--cli-input-json file://definitions/mammalia.config.json

submit-job-gammaproteobacteria:
	aws sagemaker create-processing-job \
		--processing-job-name spec2fam-$$(date +%s) \
		--cli-input-json file://definitions/gammaproteobacteria.config.json

submit-job-magnolopsida:
	aws sagemaker create-processing-job \
		--processing-job-name spec2fam-$$(date +%s) \
		--cli-input-json file://definitions/magnolopsida.config.json