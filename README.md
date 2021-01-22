## SageMaker Ground Truth Conversion Kit

Easily convert Ground Truth outputs into other industry standard formats.

## Install dependencies for test
```
pip install -e .[test]
```

## Usage

```
job_name = "gt-converter-demo-job"
converter = CocoConverter()
converter.convert_job(job_name, output_coco_json_path="output.json")
```

## Testing
```
pytest -s
```

## Formatting
```
black .
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

