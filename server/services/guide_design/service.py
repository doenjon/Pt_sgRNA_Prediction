import json
import sys
import redis
import logging
from pt_guide_design.design_guides import GuideDesigner
import tempfile
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GuideDesignService:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379))
        )
        logger.info("Connected to Redis")

    def process_job(self, job_data):
        try:
            sequence = job_data['sequence']
            result_id = job_data['resultId']

            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as temp_file:
                temp_file.write(f">input\n{sequence}\n")
                input_file = temp_file.name

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create args object with required attributes
                class Args:
                    def __init__(self):
                        self.i = input_file
                        self.o = temp_dir
                        self.p = "result"
                        self.num_guides = 5
                        self.genome = "/app/pt_guide_design/resources/genome.fa"
                        self.t = 1

                args = Args()
                
                # Run guide design
                designer = GuideDesigner(args)
                guides_df, summary_df = designer.design_guides()

                # Convert results to JSON
                results = {
                    'guides': guides_df.to_dict(orient='records'),
                    'summary': summary_df.to_dict(orient='records')[0]
                }

                return results

        except Exception as e:
            logger.error(f"Error processing job: {str(e)}", exc_info=True)
            return {'error': str(e)}
        finally:
            # Cleanup
            if 'input_file' in locals():
                os.unlink(input_file)

    def run(self):
        logger.info("Starting guide design service...")
        while True:
            try:
                # Get job from Redis queue
                _, job = self.redis.brpop('guide_design_queue')
                job_data = json.loads(job)
                logger.info(f"Processing job: {job_data['resultId']}")

                # Process the job
                results = self.process_job(job_data)

                # Send results back to Redis
                self.redis.set(
                    f"results:{job_data['resultId']}", 
                    json.dumps(results)
                )
                logger.info(f"Completed job: {job_data['resultId']}")

            except Exception as e:
                logger.error(f"Service error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    service = GuideDesignService()
    service.run() 