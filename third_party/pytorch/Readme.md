# Compile torch

Compile torch for for both Python 3.6 and Python 3.7
```
make build-torch
```

# Upload the compiled wheels to S3

```
aws s3 cp torchvision-cpp-${OUTPUT_NAME}.zip  s3://wayve-data/public/installers/torchvision-cpp-${OUTPUT_NAME}.zip
aws s3 cp torch-1.4.1-${OUTPUT_NAME}.whl s3://wayve-data/public/installers/torch-${OUTPUT_NAME}.whl

aws s3api put-object-acl --bucket wayve-data --key public/installers/torchvision-${OUTPUT_NAME}.zip --acl public-read
aws s3api put-object-acl --bucket wayve-data --key public/installers/torch-${OUTPUT_NAME}.whl --acl public-read
```
