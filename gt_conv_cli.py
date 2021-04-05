import argh
from argh import arg
from gt_conversion import CocoConverter

@arg('--job_name', help='Name of the SageMaker Ground Truth job to convert', default='d2-coco-train')
@arg('--output_coco_json_path', help='Output location for resulting json', default='output.json')
def convert_job(job_name = "gt-converter-demo-job", output_coco_json_path='output.json'):
    """
    Job conversion function
    """
    converter = CocoConverter()
    converter.convert_job(job_name, output_coco_json_path)

def main():
    parser = argh.ArghParser()
    parser.add_commands([convert_job])
    parser.dispatch()
    
if __name__ == "__main__":
    main()