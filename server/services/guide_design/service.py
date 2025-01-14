import json
import sys
import os
import redis
import logging
import tempfile

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
logger.info(f"Adding project root to path: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger.info(f"Python path: {sys.path}")

try:
    import pt_guide_design
    logger.info(f"pt_guide_design package found at: {pt_guide_design.__file__}")
    from pt_guide_design.design_guides import GuideDesigner
except ImportError as e:
    logger.error(f"Failed to import GuideDesigner. Error: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

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

            # Set model path from environment variable
            os.environ['MODEL_PATH'] = os.getenv('MODEL_PATH', '/app/packages/sgrna_scorer/resources/model.weights.h5')

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
                        self.num_guides = 25
                        self.t = 1
                        self.genome = os.getenv('GENOME_PATH', '/app/packages/pt_guide_design/resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa')

                args = Args()
                
                # Run guide design
                designer = GuideDesigner(args)
                guides_df, summary_df = designer.design_guides()

                # Convert results to JSON with the expected format
                guides = []
                for _, row in guides_df.iterrows():
                    guide = {
                        'sequence': row['targetSeq'],
                        'position': row['pos'],
                        'score': float(row['design_score']),
                        'gc_content': float((row['targetSeq'].count('G') + row['targetSeq'].count('C')) / len(row['targetSeq']) * 100),
                        'off_targets': int(row['mismatch_2'] + row['mismatch_3'] + row['mismatch_4'])
                    }
                    guides.append(guide)

                results = {
                    'guides': guides,
                    'inputSequence': sequence,
                    'summary': summary_df.to_dict(orient='records')[0]
                }

                return results

        except Exception as e:
            logger.error(f"Error processing job: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'guides': [],
                'inputSequence': sequence,
                'summary': {}
            }
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